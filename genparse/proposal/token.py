from arsenal.datastructures import LocatorMaxHeap
from arsenal.iterextras import take
from arsenal.maths import sample_dict

from genparse.proposal.trie_numba import TokenCharacterTrie
from genparse.semiring import Float
from genparse.proposal.base import Proposal


class TokenProposal(Proposal):
    """Proposal distribution that combines an `llm` and `guide`.  Let Y be the set
    of tokens and ∑ the set of characters.  We assume that llm is a distrbution
    over Y* and guide is a distribution over ∑*.

    We sample the next token y ∈ Y given ys ∈ Y* according the following
    distrbution:

      q(y | ys) ∝ p_llm(y | ys) * p_guide(φ(y) | φ(ys))

    where φ: Y* → ∑* maps token strings to character strings.

    """

    def __init__(self, *, llm, guide, K=None):
        self.K = K
        super().__init__(llm=llm, guide=guide)

    def sample_set(self, context, p_llm, draw=sample_dict):
        proposal_p = 1

        # enumerate top K tokens
        Ws = Float.chart(take(self.K, self.traverse_trie(context, p_llm)))

        # Was the distrbution truncated?  If so, use a wildcard sample to debias it.
        if self.K is not None and len(Ws) == self.K:
            # compute distribution over wildcard tokens
            P_wc = Float.chart({x: p for x, p in p_llm.items() if x not in Ws and p > 0})

            # if P_wc is non-empty, sample a wildcard token to ensure absolute continuity
            if P_wc:
                P_wc = P_wc.normalize()
                wildcard = draw(P_wc)
                proposal_p *= P_wc[wildcard]

                # compute the wildcard's weight
                p_cfg_wc = self.guide.p_next_seq(''.join(context), wildcard)

                Ws[wildcard] = (
                    p_llm[self.llm.eos if wildcard == self.guide.eos else wildcard]
                    * p_cfg_wc
                    / P_wc[wildcard]
                )

        return (Ws, proposal_p)

    def traverse_trie(self, context, p_llm):
        """
        This method will lazily enumerate the nodes in the intersection of `p_llm` and
        and the `guide` for the given context.

        Here intersection means

          guide.p(token | context) * llm.p(token | context) for tokens ∈ llm.V

        """
        assert isinstance(context, tuple), context
        assert set(context) <= self.llm.V, f'OOV detected {set(context) - self.llm.V}'

        # update the trie with the llm's distribution of next token `p_llm`.
        h = self.trie.mass_max(p_llm)

        agenda = LocatorMaxHeap()

        P = Float.chart()

        # initial conditions
        (token, node) = ('', self.trie.root)
        agenda[token, node, False] = 1
        P[node] = 1

        children = self.trie.children

        curr_priority = 1
        prev_best = 1
        while agenda:
            (token, node, done), score = agenda.popitem()

            assert score <= curr_priority
            curr_priority = score

            # terminal state
            if done:
                value = P[node] * h[node]
                assert prev_best >= value
                prev_best = value
                yield (token, value)
                continue

            # Efficiently compute guide.p(x | context + token) for x ∈ guide.V.
            # These are individual characters that are aligned with the trie.
            p = self.guide.p_next(''.join(context) + token)

            for x, y in children[node].items():
                if x is None:
                    P_y = P[node]
                    P[y] = P_y
                    agenda[token, y, True] = P_y * h[y]

                else:
                    P_y = P[node] * p[x]
                    if P_y == 0:
                        continue
                    P[y] = P_y
                    agenda[token + x, y, False] = P_y * h[y]
