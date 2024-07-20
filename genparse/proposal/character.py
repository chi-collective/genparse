from arsenal.maths import sample_dict

from genparse.proposal.trie_numba import TokenCharacterTrie
from genparse.semiring import Float
from genparse.proposal.base import Proposal

"""
Proposal distribution that combines an `llm` (token-based LM) and `guide`
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


class CharacterProposal(Proposal):
    __slots__ = ('llm', 'guide')

    def sample_set(self, context, p_llm, draw=sample_dict):
        mass = self.trie.mass_sum(p_llm)
        curr = self.trie.root
        children = self.trie.children

        token = ''

        inclusion_prob = 1  # path prefix probability
        cfg_prob = 1
        proposal_p = 1  # probability of trace

        weights = Float.chart()

        while True:
            children_curr = children[curr]
            mass_curr = mass[curr]

            p1 = Float.chart((a, mass[c] / mass_curr) for a, c in children_curr.items())
            p2 = self.guide.p_next(''.join(context) + token).trim()

            if None in p1:
                weights[token] = (mass[children_curr[None]] * cfg_prob) / inclusion_prob

            _q = (p1 * p2).trim()

            if not _q:
                break

            q = _q.normalize()

            a = draw(q)
            inclusion_prob *= q[a]
            cfg_prob *= p2[a]
            proposal_p *= q[a]

            curr = children_curr[a]
            token += a

        return (weights, proposal_p)
