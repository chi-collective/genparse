"""
Language models go here
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from genparse import Float
from arsenal.maths import sample_dict
from genparse.tokenization import decode_tokenizer_vocab


class LM:
    """We say that p: V* → [0,1] is a language model if p is a probability
    distribution over strings from some alphabet V of tokens.

    Every language model admits a left-to-right factorization `p_next`

    p(x_1 x_2 ⋯ x_T) = p_next(x_1) p_next(x_2 | x_1) ⋯ p_next(x_T | x_1 ⋯ x_{T-1}) p_next(EOS | x_1 ⋯ x_T)

    We call `p_next` the (conditional) distribution over the next token.

    """

    def __init__(self, V, eos):
        self.eos = eos
        self.V = V

    def __call__(self, ys):
        "Compute the probability of a complete string."
        raise NotImplementedError()

    def p_next(self, context):
        "Compute the (conditional) distribution over the next token given the `prefix`."
        raise NotImplementedError()

    def sample(self, ys=(), draw=sample_dict, prob=False, verbose=0, max_tokens=np.inf):
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
            ys = ys + (y,)


class LLM(LM):
    """
    This is a simple class that wraps HuggingFace transformers with support for automatic caching.
    """
    def __init__(self, model):

        if torch.cuda.is_available():
            print("GPU is available")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)

        self.model = model
        self.model.eval()   # Set the model in "evaluation mode"
        self._cache = {}

        # TODO: add the vocabulary and EOS symbols!
#        super().__init__(set(range()), eos=eos)

    def __call__(self, input_ids):
        if isinstance(input_ids, list): input_ids = torch.LongTensor([input_ids])
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            lprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        token_lprobs = torch.gather(lprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
        return np.exp(torch.sum(token_lprobs, dim=-1).item())

    def get_state(self, prefix):
        assert isinstance(prefix, tuple)
        if len(prefix) == 0:
            return None

        else:
            value = self._cache.get(prefix, None)
            if value is not None: return value

            prev_state = self.get_state(prefix[:-1])
            input_ids = torch.LongTensor([prefix[-1]]).to(self.device)

            value = self.model(input_ids=input_ids, labels=input_ids,
                               past_key_values=None if prev_state is None else prev_state.past_key_values,
                               use_cache=True)

            self._cache[prefix] = value
            return value

    # TODO: handle padding and EOS more carefully.
    def p_next(self, input_ids):
        if isinstance(input_ids, (tuple, list)): input_ids = torch.LongTensor([input_ids])
        prefix = input_ids.squeeze().tolist()
        prefix = (prefix,) if isinstance(prefix, int) else tuple(prefix)
        with torch.no_grad():
            outputs = self.get_state(prefix)
            # Calculate the log_softmax to get the log probabilities for each time step
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        #return lprobs[0,-1,:]  # return the conditional distribution of just the last token
        return probs[0,:]  # return the conditional distribution of just the last token


class NoCacheLLM(LM):
    """
    This is a simple class that wraps HuggingFace transformers.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()   # Set the model in "evaluation mode"

        # TODO: Add vocabulary and EOS
#        super().__init__(set(range()), eos=eos)

    def __call__(self, input_ids):
        raise NotImplementedError()

    def p_next(self, input_ids):
        if isinstance(input_ids, (tuple, list)): input_ids = torch.LongTensor([input_ids])
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            lprobs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return lprobs[0,-1,:]  # return the conditional distribution of just the last token


class GreedilyTokenizedLLM(LM):

    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name)
        self.model = LLM(self._model)
        self._decode = decode_tokenizer_vocab(self.tokenizer)
        super().__init__(V = set(self._decode), eos = self.tokenizer.eos_token)

    def __call__(self, xs):
        return self.model(self.tokenizer.encode(xs))

    def p_next(self, xs, top=None):
        # TODO: support token healing and/or constrained generation to get a
        # valid token sequence; see `p_next_healing`.
        assert isinstance(xs, str)
        tokens = self.tokenizer.encode(xs)
        _p = self.model.p_next(tokens).cpu().numpy()
        if top is None:
            top_p = _p.argsort()
        else:
            top_p = _p.argsort()[-top:]
        pp = Float.chart()
        for i in reversed(top_p):
            pp[self._decode[i]] = _p[i]
        if top is None:
            return pp
        else:
            return pp.normalize()

    # TODO: why isn't this inherited from the LM base class?
    def sample(self, ys='', draw=sample_dict, prob=False, verbose=0, max_tokens=np.inf, join=str.__add__):
        P = 1.0
        t = 0
        while True:
            p = self.p_next(ys)
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

#    def p_next_healing(self, xs, top=10):
#        # TODO: support token healing and/or hindsight sampling to get a valid token sequence
#        assert isinstance(xs, str)
#        tokens = self.tokenizer.encode(xs)
#        # token healing will take all but the last token and then resample the last one
#        # since it might be a partial token.
#        print([(t, self.tokenizer.decode([t])) for t in tokens])
#        complete = self.tokenizer.decode(tokens[:-1])
#        token_prefix = xs[len(complete):]
#        tokens = tokens[:-1]
#        _p = self.model.p_next(tokens).numpy()
#        pp = Float.chart()
#        for i in reversed(_p.argsort()):
#            x = self.tokenizer.decode([i])
#            if x.startswith(token_prefix):
#                pp[x] = _p[i]
#                if len(pp) > top: break
#        return pp

class AsyncGreedilyTokenizedLLM(LM):
    """
    This is a simple class which wraps HFPPL CachedCausalLMs. 
    Caching is done by HFPPL.
    """
    def __init__(self, llm, tokenizer):
        """
        Args:
            llm (hfppl.llms.CachedCausalLM): The HFPPL CachedCausalLM.
            tokenizer: The underlying HuggingFace tokenizer.
        """
        self.tokenizer = tokenizer
        self._model = llm
        self._decode = decode_tokenizer_vocab(self.tokenizer)
        super().__init__(V = set(self._decode), eos = self.tokenizer.eos_token)

    def __call__(self, xs):
        return self.model(self.tokenizer.encode(xs))

    async def p_next(self, xs, top=None):
        return await self._p_next(xs, top=top)

    async def _p_next(self, xs, top=None):
        assert isinstance(xs, str)
        tokens = self.tokenizer.encode(xs)

        _logp = await self._model.next_token_logprobs(tokens)
        _p = np.exp(_logp)

        if top is None:
            top_p = _p.argsort()
        else:
            top_p = _p.argsort()[-top:]
        pp = Float.chart()
        for i in reversed(top_p):
            pp[self._decode[i]] = _p[i]
        if top is None:
            return pp
        else:
            return pp.normalize()


from functools import lru_cache
@lru_cache(None)
def make_mock_llm(**kwargs):
    from genparse.util import hf_tokenizer
    H = hf_tokenizer(**kwargs)
    return MockLLM(V = H.decode, eos = H.eos)


class MockLLM(LM):
    """
    Uniform distribution over next token; used for testing.
    """
    def __init__(self, V, eos):
        n = len(V)
        self._p = Float.chart({w: 1/n for w in V})
        super().__init__(
            eos = eos,
            V = V,
        )
    def p_next(self, _):
        return self._p

    def __call__(self, x):
        assert x[-1] == self.eos
        return (1/len(self.V))**len(x)
