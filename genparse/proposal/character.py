from arsenal import colors
from arsenal.maths import sample_dict

from genparse.proposal.trie_numba import TokenCharacterTrie
from genparse.semiring import Float
from genparse.proposal.base import Proposal


class CharacterProposal(Proposal):
    """Proposal distribution that combines an `llm` (token-based LM) and `guide`
    (character-based LM).

    The way that samples are generated is that we
    (1) materialize the next-token distribution `llm.p_next(context)`
    (2) convert it into a character-level trie augmented with an end-of-token marker.
    (3) sample a path in the trie (starting at its root) which takes the local
        product of the trie distribution and the guide, excluding the
        end-of-token.
    (4) given the path, we then sample an end-of-token anywhere along the path.

    The reason why we like this proposal distribution is its efficiency: in
    practice, `p_llm` is one big batched evaluation, that is given by a blackbox
    model, and `p_guide` is a character-level LM.  Although, any given call to
    p_guide is fast, calling it for every token is very slow - even with GPU
    parallelism.  This proposal distrbution avoid making a huge number of calls
    to p_guide (as in `CharAlignedCFGLM`) by *sampling* paths in the
    character-trie rather than *enumerating* them.

    We could probably improve this generative procees by collapsing the
    post-path sampling of exits, but it would probably require the cost that we
    are trying to avoid!  (That is probably deeply connected with
    `CharAlignedCFGLM`, but we haven't worked out the precise connection.)

    """

    __slots__ = (
        'llm',
        'guide',
    )

    def __init__(self, *, llm, guide):
        self.llm = llm
        self.guide = guide

        # Filter LLM tokens that are illegal under the cfg
        words = {word for word in llm.V if set(word) <= self.guide.V or word == llm.eos}

        self.trie = TokenCharacterTrie(
            words, encode=llm._encode, old_eos=llm.eos, new_eos=guide.eos
        )

    async def sample_next_token(
        self,
        prompt,
        context,
        draw=sample_dict,
        p_llm=None,
        **kwargs,
    ):
        """
        Proposes a token and incremental weight update.

        Args:
          - prompt : The LLM prompt.
          - context : The previous generated tokens.
          - correct_weights : Whether to correct the importance weights with RAVI.
                false leads to improperly weighted samples.
          - p_llm: Provide the model with pre-computed p_llm. Since for VLLM, p_llm is computed
                for all particles altogether. We directly pass the corresponding p_llm to
                the proposal of each particle.
        Returns:
          - token : Proposed LLM token.
          - proposal_p :
          - weight_update : Incremental SMC weight update.

        This function samples a token from the trie and computes the incremental weight update.

        The following procedure, justified using RAVI, gives the way we sample a token and compute the incremental SMC weight update.

            1. Sample a subset $S$ of the token vocabulary by sampling a path through the trie.
            2. Compute *unnormalized target* $p(x)$ of each $x \in S$ according to $p_\text{LLM}(x)p_\text{CFG}(x)$.
                * $p_\text{LLM}(x)$ is given from the mass at the leaf of the trie;
                * $p_\text{CFG}(x)$ is given as the product of the next character distributions up to that point in the path
            3. Compute (local) weight $w(x)$ of each token as $\frac{p(x)}{\Pr(x \in S)}$ where $\Pr(x \in S)$ is the *inclusion probability*.
                * $\Pr(x \in S)$ in the character proposal is given as the probability of the path prefix up to $x$.
            4. Renormalize the weights of the tokens in $S$ and sample one of them.
            5. Set the incremental SMC weight update $w^\prime(x) = \sum_{x \in S} w(x)$

        """

        if p_llm is None:
            p_llm = await self.llm.p_next_async(prompt + context)

        mass = self.trie.mass_sum(p_llm)
        curr = self.trie.root
        children = self.trie.children

        path = []
        inclusion_prob = 1  # path prefix probability
        cfg_prob = 1
        proposal_p = 1  # probability of trace

        weights = Float.chart()

        while True:
            children_curr = children[curr]
            mass_curr = mass[curr]

            p1 = Float.chart((a, mass[c] / mass_curr) for a, c in children_curr.items())

            p2 = self.guide.p_next(''.join(context) + ''.join(path))

            if None in p1:
                weights[''.join(path)] = (
                    mass[children_curr[None]] * cfg_prob
                ) / inclusion_prob

            _q = (p1 * p2).trim()

            if not _q:
                break

            q = _q.normalize()

            a = draw(q)
            inclusion_prob *= q[a]
            cfg_prob *= p2[a]
            proposal_p *= q[a]

            curr = children_curr[a]
            path.append(a)

        normalized_weights = weights.normalize()
        token = draw(normalized_weights)
        proposal_p *= normalized_weights[token]
        weight_update = weights.sum()

        return (token, proposal_p, weight_update)
