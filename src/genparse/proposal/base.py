from genparse.proposal.trie_numba import TokenCharacterTrie
from arsenal.maths import sample_dict
import numpy as np
from genparse.semiring import Float
from genparse.trace import TraceSWOR
from arsenal import colors
import warnings
import asyncio


class Proposal:
    r"""
    Abstract class for proposal distributions which sample a set of tokens S and proposes a token by sampling one from S.

    For properly weighted inference, a token x should be sampled from S proportionally to the *local weight* of x.

    The local weight of x ∈ S corresponds to:

        $p_\text{LLM}(x | prompt, context)p_\text{WCFG}(x | context) / \Pr(x ∈ S)$

    where $\Pr(x ∈ S)$ is the inclusion probability of x in S.
    """

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

        self.eos = self.trie.new_eos

    def sample_set(self, context, prompt=None, p_next_llm=None, **kwargs):
        raise NotImplementedError('Subclasses must implement the sample_set method')

    async def sample_next_token(
        self,
        context,
        p_llm=None,
        prompt=None,
        draw=sample_dict,
        properly_weight=True,
        **kwargs,
    ):
        """Proposes a token and compute its RAVI incremental weight update."""
        if p_llm is None:
            assert prompt is not None, (
                'Must provide prompt to obtain LM next token probabilities. '
                'Alternatively, provide the next token probabilities via the `p_llm` argument.'
            )
            p_llm = await self.llm.p_next_async(prompt + context)

        (weights, p) = self.sample_set(context, p_llm=p_llm, draw=draw, **kwargs)

        if weights:
            probs = weights.normalize()
            unit = draw(probs)
            p *= probs[unit]
            if properly_weight:
                log_weight_update = np.log(weights.sum())
            else:
                log_weight_update = 0
        else:
            warnings.warn('No possible next units found. Killing particle.')
            # if there are no possible next units, kill the particle
            unit = self.eos
            log_weight_update = -np.inf

        return (unit, p, log_weight_update)

    def sample_next_token_sync(self, *args, **kwargs):
        "Synchronous version of `sample_next_token`."
        return asyncio.run(self.sample_next_token(*args, **kwargs))

    def sample(self, prompt=(), max_tokens=float('inf'), verbosity=0, draw=sample_dict):
        """Performs sequential importance sampling by sequentially sampling tokens from the proposal distribution."""
        context = ()
        log_W = 0
        P = 1
        t = 0
        while True:
            t += 1
            if t <= max_tokens:
                (token, proposal_p, weight_update) = self.sample_next_token_sync(
                    prompt=prompt,
                    context=context,
                    draw=draw,
                )
            else:
                token = self.guide.eos
                weight_update = 0
                proposal_p = 1
            log_W += weight_update
            P *= proposal_p
            if self.guide.eos == token:
                break
            if verbosity > 0:
                print(colors.cyan % token, end=colors.magenta % '|')
            context = context + (token,)
        if verbosity > 0:
            print()
        return (context, P, np.exp(log_W))

    def enumerate_traces(self, prompt, context):
        """
        This function uses program tracing and sampling without replacement to compute

            E_{(x,w) ~ q'}[ δ(x, x') * w ] = E_{(x,S) ~ q}[ δ(x, x') * w(x,S) ]
                                        = Σ_{x,S} δ(x, x') * q(x,S) * w(x,S)

        for each x' in V.

        Its use is to check whether our proposal satisfies properties like proper weighting through exact enumeration.
        """
        tracer = TraceSWOR()
        P = Float.chart()
        # sample without replacement until all traces have been exhausted
        while tracer.root.mass > 0:
            with tracer:
                (s, q, w) = self.sample_next_token_sync(
                    draw=tracer, prompt=prompt, context=context
                )
                P[s] += np.exp(w) * q
        return (P, tracer)
