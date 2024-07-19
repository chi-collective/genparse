from genparse.proposal.trie_numba import TokenCharacterTrie
from arsenal.maths import sample_dict
import numpy as np


class Proposal:
    def __init__(self, llm, guide):
        self.llm = llm
        self.guide = guide

        # Filter LLM tokens that are illegal under the cfg
        words = {word for word in llm.V if set(word) <= self.guide.V or word == llm.eos}

        # Augment the guide's character vocabulary to avoid to avoid OOV issues
        self.guide.V |= {w for word in llm.V for w in word}

        self.trie = TokenCharacterTrie(
            words, encode=llm._encode, old_eos=llm.eos, new_eos=guide.eos
        )

    def sample(self, context, p_llm=None, prompt=None, draw=sample_dict, **kwargs):
        """Proposes a token and compute its RAVI incremental weight update."""
        if p_llm is None:
            assert prompt is not None, (
                'Must provide prompt to obtain LM next token probabilities. '
                'Alternatively, provide the next token probabilities via the `p_llm` argument.'
            )
            p_llm = self.llm.p_next(prompt + context)

        (weights, p) = self.sample_set(context, p_llm=p_llm, **kwargs)

        if weights:
            probs = weights.normalize()
            unit = draw(probs)
            p *= probs[unit]
            weight_update = weights.sum()
        else:
            # if there are no possible next units, kill the particle
            unit = '💀'
            weight_update = 0

        return (unit, p, np.log(weight_update))
