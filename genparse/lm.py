"""
Language models go here
"""

import asyncio
import numpy as np
import torch
from arsenal.maths import sample_dict

from genparse.semiring import Float
from genparse.tokenization import decode_tokenizer_vocab
from genparse.backends.vllm import vllmpplLLM
from genparse.util import top_p_filter


class LM:
    r"""We say that $p\colon V^* \to [0,1]$ is a language model if $p$ is a probability
    distribution over strings from some alphabet $V$ of tokens.

    Every language model admits a left-to-right factorization:

    $$
    p(x_1 x_2 \cdots x_T) = p(x_1 \mid \varepsilon) p(x_2 \mid x_1) \cdots p(x_T \mid x_1 \cdots x_{T-1}) p(\mathrm{EOS} \mid x_1 \cdots x_T)
    $$

    Arguments:

      - `V`: a vocabulary of symbols

      - `eos`: a distinguished end of sequence symbol

      - `p_next(xs)`: $p(\cdot \mid x_1 \cdots x_T)$ is provided by subclasses.

    """

    def __init__(self, V, eos):
        self.eos = eos
        self.V = V

    def __call__(self, context):
        "Compute the probability of a complete string."
        # return np.exp(self.logp(ys))
        assert context[-1] == self.eos
        P = 1
        for i, y in enumerate(context):
            assert y in self.V, y
            p = self.p_next(context[:i])
            P *= p[y]
            if P == 0:
                break
        return P

    def logp(self, context):
        "Compute the probability of a complete string."
        assert context[-1] == self.eos
        return sum(self.logp_next(context[:i])[y] for i, y in enumerate(context))

    def logp_next(self, context):
        "Compute the log conditional distribution over the next token given the `prefix`."
        raise NotImplementedError()

    def p_next(self, context):
        "Compute the conditional distribution over the next token given the `prefix`."
        raise NotImplementedError()

    async def p_next_async(self, context):
        "Compute the (conditional) distribution over the next token given the `prefix`."
        return self.p_next(context)

    def p_next_seq(self, context, extension):
        """
        Compute `p(extension | context)` where `extension` is a sequence with |extension| > 1.
        """
        assert len(extension) >= 1
        P = 1
        for i in range(len(extension)):
            p = self.p_next(context + extension[:i])
            P *= p[extension[i]]
        return P

    def clear_cache(self):  # pragma: no cover
        pass

    def sample(
        self,
        ys=(),
        draw=sample_dict,
        prob=True,
        verbose=0,
        max_tokens=np.inf,
        join=lambda ys, y: ys + (y,),
    ):
        assert isinstance(ys, tuple), ys
        P = 1.0
        t = 0
        while True:
            p = self.p_next(ys).normalize()
            y = draw(p) if t <= max_tokens else self.eos
            P *= p[y]
            t += 1
            if verbose:
                if y == self.eos:
                    print()
                else:
                    print(y, end='')
            if y == self.eos:
                return (ys, P) if prob else ys
            ys = join(ys, y)


class LLM(LM):
    """
    This is a simple class that wraps HuggingFace transformers with support for automatic caching.
    """

    def __init__(self, model, V, eos):
        super().__init__(V=V, eos=eos)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)

        self.model = model
        self.model.eval()  # Set the model in "evaluation mode"
        self._cache = {}

    def __call__(self, context):
        return np.exp(self.logp(context))

    def logp(self, context):
        input_ids = context
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor([input_ids]).squeeze()
        if input_ids[0] != self.model.config.bos_token_id:
            input_ids = torch.cat(
                [torch.LongTensor([self.model.config.bos_token_id]), input_ids]
            )
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            lprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        token_lprobs = torch.gather(lprobs, 1, input_ids[1:].unsqueeze(-1)).squeeze(-1)
        return torch.sum(token_lprobs, dim=-1).item()

    def clear_cache(self):
        self._cache.clear()

    async def next_token_logprobs(self, context):
        return self.p_next(context).log()

    def get_state(self, prefix):
        assert isinstance(prefix, tuple)
        if len(prefix) == 0:
            return None

        else:
            value = self._cache.get(prefix, None)
            if value is not None:
                return value

            prev_state = self.get_state(prefix[:-1])
            input_ids = torch.LongTensor([prefix[-1]]).to(self.device)

            value = self.model(
                input_ids=input_ids,
                labels=input_ids,
                past_key_values=(
                    None if prev_state is None else prev_state.past_key_values
                ),
                use_cache=True,
            )

            self._cache[prefix] = value
            return value

    def p_next(self, context):
        return np.exp(self.logp_next(context).cpu())

    def logp_next(self, context):
        if isinstance(context, (tuple, list)):
            context = torch.LongTensor([context])
        prefix = context.squeeze().tolist()
        prefix = (prefix,) if isinstance(prefix, int) else tuple(prefix)
        BOS = self.model.config.bos_token_id
        prefix = (BOS,) + prefix if len(prefix) == 0 or prefix[0] != BOS else prefix
        with torch.no_grad():
            outputs = self.get_state(prefix)
            # Calculate the log_softmax to get the log probabilities for each time step
            probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        # return lprobs[0,-1,:]  # return the conditional distribution of just the last token
        return probs[0, :]  # return the conditional distribution of just the last token


class TokenizedLLM(LM):
    """
    This is a simple class which wraps a token LLM with a tokenizer.
    """

    def __init__(self, tokenizer, model, batch_size, temperature=1, top_p=None):
        self.tokenizer = tokenizer
        self._model = model
        self._model.batch_size = batch_size
        self._decode = decode_tokenizer_vocab(self.tokenizer)
        self._encode = {x: i for i, x in enumerate(self._decode)}
        self.temperature = temperature
        self.top_p = top_p
        super().__init__(V=set(self._decode), eos=self.tokenizer.eos_token)

    def encode_prompt(self, prompt):
        "Encode `prompt` as a tuple of tokens (each a string)."
        return tuple(self._decode[i] for i in self.tokenizer.encode(prompt))

    def __call__(self, context):
        assert isinstance(context, tuple), '`context` must be explicitly tokenized'
        assert set(context) <= self.V, f'OOVs detected: {set(context) - self.V}'
        assert (
            context[-1] == self.eos
        ), f'Context must end with eos ({self.eos!r}); got {context = }.'
        if self.temperature == 1 and self.top_p is None:
            return self._model([self._encode[x] for x in context])
        else:
            return super().__call__(context)

    def clear_cache(self):
        return self._model.clear_cache()

    async def next_token_logprobs(self, context):
        p = await self.p_next_async(context)
        return p.map_values(np.log)

    def p_next(self, context, _logp=None):
        return asyncio.run(self.p_next_async(context, _logp=None, return_logp=False))

    def logp_next(self, context, _logp=None):
        return asyncio.run(self.p_next_async(context, _logp=None, return_logp=True))

    async def p_next_async(self, context, _logp=None, return_logp=False):
        # For vllm, we need to provide the log probabilities, and
        # _logp is provided by the vllm centralized step function

        assert (
            not isinstance(self._model, vllmpplLLM) or _logp is not None
        ), 'vLLM requires `_logp` to be passed.'

        if _logp is None:
            assert isinstance(
                context, tuple
            ), 'API change; `context` must be explicitly tokenized'
            assert set(context) <= self.V, f'OOVs detected: {set(context) - self.V}'

            tokens = [self._encode[x] for x in context]
            _logp = await self._model.next_token_logprobs(tokens)

        _logp = _logp.cpu().numpy() if hasattr(_logp, 'cpu') else _logp

        # Below, we apply post-processing operations on the conditional
        # probabilities, such as temperature and top-p filters.

        if self.temperature != 1:
            _logp = _logp / self.temperature

        exp_dtype = np.float64 if np.any(_logp < -100) else _logp.dtype
        assert np.all(_logp > -700), 'log probabilities too low, will cause underflow'
        _p = np.exp(_logp, dtype=exp_dtype)
        _p /= _p.sum()

        if self.top_p is not None:
            _p = top_p_filter(_p.copy(), self.top_p)

        if return_logp:
            return LazyProb(np.log(_p), self._encode, self._decode)
        else:
            return LazyProb(_p, self._encode, self._decode)


class VirtualTokenizedLLM(TokenizedLLM):
    def __init__(self, vllm_engine):
        self.llm_engine = vllm_engine
        self.tokenizer = self.llm_engine.get_tokenizer()
        super().__init__(tokenizer=self.tokenizer, model=vllm_engine, batch_size=None)

    @classmethod
    def from_name(cls, model_name):
        from vllm import LLMEngine, EngineArgs

        return cls(
            LLMEngine.from_engine_args(  # seed not used since we are not sampling with vllm
                EngineArgs(model=model_name, tokenizer=model_name, seed=0)
            )
        )

    # TODO: support the following methods, for easier debugging

    def __call__(self, **kwargs):
        raise NotImplementedError('Cannot call VirtualTokenizedLLM directly')

    def clear_cache(self):
        pass

    async def next_token_logprobs(self, **kwargs):
        raise NotImplementedError()

    def p_next(self, **kwargs):
        raise NotImplementedError()

    def logp_next(self, **kwargs):
        raise NotImplementedError()

    async def p_next_async(self, **kwargs):
        raise NotImplementedError()


#    def p_next_healing(self, context, top=10):
#        # TODO: support token healing and/or hindsight sampling to get a valid token sequence
#        assert isinstance(context, str)
#        tokens = self.tokenizer.encode(context)
#        # token healing will take all but the last token and then resample the last one
#        # since it might be a partial token.
#        print([(t, self.tokenizer.decode([t])) for t in tokens])
#        complete = self.tokenizer.decode(tokens[:-1])
#        token_prefix = context[len(complete):]
#        tokens = tokens[:-1]
#        _p = self.model.p_next(tokens).numpy()
#        pp = Float.chart()
#        for i in reversed(_p.argsort()):
#            x = self.tokenizer.decode([i])
#            if x.startswith(token_prefix):
#                pp[x] = _p[i]
#                if len(pp) > top: break
#        return pp


class LazyProb:
    """
    This class is used to efficiently associate string with the indices of LLM's
    tokens distribution over next tokens.
    """

    def __init__(self, _p: torch.tensor, encode: dict[str, int], decode: list[str]):
        self._p = _p
        self._encode = encode
        self._decode = decode

    def normalize(self):
        return LazyProb(
            _p=self._p / self._p.sum(),
            encode=self._encode,
            decode=self._decode,
        )

    def keys(self):
        return self._decode

    def values(self):
        return self._p

    def items(self):
        return zip(self._decode, self._p)

    def __getitem__(self, token: str) -> float:
        i = self._encode.get(token)
        return self._p[i] if i is not None else 0

    def materialize(self, top=None):
        _p = self._p
        _decode = self._decode

        top_p = _p.argsort() if top is None else _p.argsort()[-int(top) :]

        pp = Float.chart()
        for i in reversed(top_p):
            pp[_decode[i]] = _p[i]

        return pp if top is None else pp.normalize()

    def __repr__(self):
        return repr(self.materialize())


class MockLLM(LM):
    """
    Uniform distribution over next token; used for testing.
    """

    def __init__(self, V, eos, _p=None):
        n = len(V)
        self._p = np.array([1 / n for _ in range(len(V))]) if _p is None else _p
        self._logp = np.log(self._p)
        self._decode = list(V)
        self._encode = {x: i for i, x in enumerate(self._decode)}
        super().__init__(eos=eos, V=set(V))

    def p_next(self, context):
        assert isinstance(context, tuple)
        assert set(context) <= self.V, f'OOVs detected: {set(context) - self.V}'
        return LazyProb(self._p, self._encode, self._decode)

    def logp_next(self, context):
        assert isinstance(context, tuple)
        assert set(context) <= self.V, f'OOVs detected: {set(context) - self.V}'
        return LazyProb(self._logp, self._encode, self._decode)

    async def next_token_logprobs(self, _):
        return self._logp
