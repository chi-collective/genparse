"""
Sample a sequence by repeatedly sampling from a truncated version of the local product of experts.

Warning: This method does not produce properly weighted samples.
"""

import torch.nn.functional as F
from arsenal.iterextras import take
import numpy as np

from genparse.tokenization import decode_tokenizer_vocab
from genparse.proposal import TokenProposal
from genparse import Float
from genparse.lm import LM
from genparse.steer import ParticleApproximation

from vllm import SamplingParams


class MinimalLM(LM):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._decode = decode_tokenizer_vocab(self.tokenizer)
        self.eos_token_id = self.tokenizer.eos_token_id
        super().__init__(V=set(self._decode), eos=self.tokenizer.eos_token)

    @classmethod
    def from_vllm(cls, vllm_llm):
        return cls(vllm_llm.get_tokenizer())


class LocalMasker:
    def __init__(self, minimal_lm, guide, K=None):
        self.llm = minimal_lm
        self.guide = guide
        self.token_proposal = TokenProposal(llm=self.llm, guide=self.guide, K=K)
        self.K = K

    def __call__(self, token_ids, logits):
        probs = F.softmax(logits, dim=0).cpu()
        p_llm = Float.chart()
        for i in range(len(logits)):
            p_llm[self.llm._decode[i]] = probs[i]

        # lazily enumerate the top K nodes in the intersection of `p_llm` and the `guide`
        # p_next_[token] = p_cfg(token | context) * p_llm(token | prompt, context)
        context = tuple(self.llm._decode[i] for i in token_ids)
        p_next_ = Float.chart(
            take(self.K, self.token_proposal.traverse_trie(context, p_llm))
        )

        if not p_next_:
            raise ValueError('No possible next tokens', p_next_)

        for i in range(len(logits)):
            # coerce EOS token
            if i == self.llm.eos_token_id:
                _p = p_next_[self.guide.eos]
            else:
                _p = p_next_[self.llm._decode[i]]

            logits[i] = np.log(_p) if _p > 0 else -np.inf

        return logits


class Particle:
    def __init__(self, context, token_ids, weight, finished):
        self.context = context
        self.context_ids = token_ids
        self.weight = weight
        self.finished = finished


class LocalPOESampler:
    """
    Samples sequences by repeatedly sampling from the truncated local product of experts distribution.

    Warning: Probabilistically incorrect -- samples will be biased.

    Args:

    - vllm_llm (vllm.LM) : VLLM language model
    - guide (genparse.LM) : Character-level guide
    - K : Number of top-K tokens in the local product of experts to enumerate.
        When K = None, the full local product of experts is enumerated at each time-step.


    Example usage:
    >>> from vllm import LLM
    >>> vllm_llm = LLM(model = 'codellama/CodeLlama-7b-Instruct-hf')
    >>> from genparse.util import lark_guide
    >>> guide = lark_guide(grammar)
    >>> from genparse.experimental.steer_local import LocalPOESampler
    >>> sampler = LocalPOESampler(vllm_llm, guide, K = 5)
    >>> approx = sampler.run_inference(
            prompt = prompt,
            n_particles = 5,
            max_tokens = 100,
            seed = 0
        )
    """

    def __init__(self, vllm_llm, guide, K):
        self.vllm_llm = vllm_llm
        self.guide = guide
        self.llm = MinimalLM.from_vllm(vllm_llm)
        self.masker = LocalMasker(self.llm, self.guide, K=K)

    def run_inference(self, prompt, n_particles, max_tokens, seed):
        sampling_params = SamplingParams(
            n=n_particles,
            seed=seed,
            logits_processors=[self.masker],
            max_tokens=max_tokens,
        )

        outputs = self.vllm_llm.generate([prompt], sampling_params)

        assert len(outputs) == 1

        particles = []
        for seq in outputs[0].outputs:
            context = [
                self.llm._decode[i] if not i == self.llm.eos_token_id else self.guide.eos
                for i in seq.token_ids
            ]

            particles.append(
                Particle(
                    context=context,
                    token_ids=seq.token_ids,
                    weight=0,
                    finished=seq.finish_reason == 'stop',
                )
            )

        return ParticleApproximation(particles)
