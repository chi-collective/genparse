"""
Language models go here
"""

import numpy as np
from functools import lru_cache

from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# TODO: be careful with log probs vs probs vs weights vs log weights.

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


# Note that there is in theory a class `GlobalProduct(LM)` that implements the
# globally normalized product of experts.  Unfortunately, we cannot implement it
# efficiently in most cases of interest.  Some special cases exist for finite
# LMs, finite-state LMS, and (finite state + CFG LMs).  What we will develop is
# an approximate LM that gives us an approximation to this interface.  It will
# give us importance-weighted samples that provide a statistcally consistent
# estimate of the distribution (as the number of samples used to form it go to
# infinity), assuming that the proposal distribution has support everywhere the
# the global product does.

# TODO: should there be an `EnergyBasedLM` class to represent unnormalized models?

# TODO: test the LocalProduct qith with the `run` method - currently blocked on
# an efficient way to compute the weights.

#____________________________________________________________________________________
#


class TokenGPT2(LM):
    """
    This is a simple class that wraps HuggingFace transformers with support for automatic caching.
    """
    def __init__(self, gpt_model):
        self.model = gpt_model
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
        prefix = tuple(input_ids.squeeze().tolist())
        with torch.no_grad():
            outputs = self.get_state(prefix)
            # Calculate the log_softmax to get the log probabilities for each time step
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        #return lprobs[0,-1,:]  # return the conditional distribution of just the last token
        return probs[0,:]  # return the conditional distribution of just the last token


class NoCacheGPT(LM):
    """
    This is a simple class that wraps HuggingFace transformers.
    """
    def __init__(self, gpt_model):
        self.model = gpt_model
        self.model.eval()   # Set the model in "evaluation mode"

    # TODO: handle padding and EOS more carefully.
    def p_next(self, input_ids):
        if isinstance(input_ids, (tuple, list)): input_ids = torch.LongTensor([input_ids])
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            lprobs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return lprobs[0,-1,:]  # return the conditional distribution of just the last token


class TokenizingLM:

    def __init__(self, tokenizer, model):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, x):
        return self.model(self.tokenizer.encode(x))

    def p_next(self, x):
        # warning: misalignment is might occur here!
        p = self.model.p_next(self.tokenizer.encode(x))

        assert len(p) == self.tokenizer.vocab_size
        s = [self.tokenizer.decode([k]) for k in range(len(p))]

        # XXX: apparently, multiple tokens in the mapping (token -> substring)
        # can point to the same substring!
        P = Counter()
        for k in range(len(p)):
            P[s[k]] += p[k].item()

        return P
