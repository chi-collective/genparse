"""
Language models go here
"""

import numpy as np
import torch
from functools import lru_cache
from collections import Counter


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
        prefix = input_ids.squeeze().tolist()
        prefix = (prefix,) if isinstance(prefix, int) else tuple(prefix)
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

    def __call__(self, input_ids):
        raise NotImplementedError()

    # TODO: handle padding and EOS more carefully.
    def p_next(self, input_ids):
        if isinstance(input_ids, (tuple, list)): input_ids = torch.LongTensor([input_ids])
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            lprobs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return lprobs[0,-1,:]  # return the conditional distribution of just the last token
