"""
Language models go here
"""

import numpy as np
import torch
from functools import lru_cache
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from genparse import Float


# In the most abstract case, an LM is a probability distribution over strings
# from some alphabet that sums to one.  The `p_next` part is an additional part
# that depends on a specific factorization (many factorizations exist by the
# chain rule of probability, nuances aside).
class LM:

    def __call__(self, ys):
        "Compute the probability of a complete string"
        raise NotImplementedError()

    def p_next(self, prefix):
        "Compute the distribution over the next token given the `prefix`."
        raise NotImplementedError()


class LLM(LM):
    """
    This is a simple class that wraps HuggingFace transformers with support for automatic caching.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()   # Set the model in "evaluation mode"

    def __call__(self, input_ids):
        if isinstance(input_ids, list): input_ids = torch.LongTensor([input_ids])
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            lprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        token_lprobs = torch.gather(lprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
        return np.exp(torch.sum(token_lprobs, dim=-1).item())

    @lru_cache(None)
    def get_state(self, prefix):
        assert isinstance(prefix, tuple)
        if len(prefix) == 0:
            return None
        else:
            prev_state = self.get_state(prefix[:-1])
            #input_ids = torch.LongTensor([list(prefix)])
            input_ids = torch.LongTensor([prefix[-1]])
            return self.model(input_ids=input_ids,
                              labels=input_ids,
                              past_key_values=None if prev_state is None else prev_state.past_key_values,
                              use_cache=True)

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

    def __call__(self, input_ids):
        raise NotImplementedError()

    # TODO: handle padding and EOS more carefully.
    def p_next(self, input_ids):
        if isinstance(input_ids, (tuple, list)): input_ids = torch.LongTensor([input_ids])
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            lprobs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return lprobs[0,-1,:]  # return the conditional distribution of just the last token


class GreedilyTokenizedLLM:

    def __init__(self, name):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(name)
        self.model = LLM(self._model)

    def __call__(self, xs):
        return self.model(self.tokenizer.encode(xs))

    def p_next(self, xs, top=None):
        # TODO: support token healing and/or constrained generation to get a valid token sequence
        assert isinstance(xs, str)
        tokens = self.tokenizer.encode(xs)
        _p = self.model.p_next(tokens).numpy()
        if top is None:
            top_p = _p.argsort()
        else:
            top_p = _p.argsort()[-top:]
        pp = Float.chart()
        for i in reversed(top_p):
            x = self.tokenizer.decode([i])
            pp[x] = _p[i]
        return pp

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
