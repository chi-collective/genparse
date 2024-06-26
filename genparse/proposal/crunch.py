from arsenal.datastructures.pdict import pdict
from collections import namedtuple
from arsenal.iterextras import head_iter, take
from arsenal import colors
from time import time
from genparse import Float


Item = namedtuple('Item', 'ps, xs, ys')


from genparse.proposal import TokenProposal


class Crunching:
    """A* enumeration of sequences subject to constraints.

    Crunching (N-Best approximation) is a technique used to marginalize over
    latent variables (typically aligments) in machine translation.  People often
    credit May and Knight (2006) in the context of machine translation.

    """

    def __init__(self, *, llm, guide):
        self.llm = llm
        self.guide = guide
        self.T = TokenProposal(llm=llm, guide=guide, K=None)

    def posterior_enumerate(self, depth):
        Q = pdict()

        it = head_iter(
            self._iter_p_next(Item(1, '', (self.llm.tokenizer.bos_token,)))
        )  # XXX: empty string didn't work
        Q[it] = -1

        while Q:
            iterator = Q.pop()

            if iterator.done:
                continue
            item = next(iterator)
            print('.', end='')

            # TODO: both models must generate their respective EOS, not just the guide...
            if item.ys[-1] == self.guide.eos:
                if self.guide(item.xs):
                    yield item
                else:
                    continue

            if len(item.ys) <= depth:
                extend_iter = head_iter(self._iter_p_next(item))
                if not extend_iter.done:
                    Q[extend_iter] = -extend_iter.head.ps

            if not iterator.done:
                Q[iterator] = -iterator.head.ps

    # simplified version doesn't not benefit for fast guide.p_next computation
    #    def _____iter_p_next(self, item):
    #        ps, xs, ys = item
    #
    #        #distribution = llm.p_next(ys)
    #        distribution = self.llm.p_next(''.join(ys))
    #
    #        order = distribution._p.argsort()
    #
    #        for i in reversed(order):
    #            p = distribution._p[i]
    #
    #            if p == 0: break
    #            x = distribution._decode[i]
    #            xsx = xs + x
    #
    #            z = self.guide.p_next(xsx).trim()
    #            if len(z) == 0: continue
    #
    #            y = x    # it's already a character string
    #
    #            yield Item(
    #                xs = xs + x,
    #                ps = ps * p,
    #                ys = ys + (y,),
    #            )

    def _iter_p_next(self, item):
        """
        This method will lazily enumerate the nodes in the intersection of `llm` and
        and the `guide` for the given context.

        Here intersection means

          guide.p(token | context) * llm.p(token | context) for tokens ∈ llm.V

        """

        T = self.T
        p_llm = self.llm.p_next(''.join(item.ys))
        T._update_leaves(p_llm)

        mass = T.mass.copy()

        # Update internal nodes for our A* heuristic
        jump = T.jump
        for node in T.ordering:
            m = 0
            for child in jump[node]:
                m = max(m, mass[child])
            mass[node] = m

        agenda = pdict()
        P = Float.chart()

        # initial conditions
        (token, node) = ('', T.root)
        agenda[token, node] = 0
        P[node] = 1

        while agenda:
            (token, node) = agenda.pop()

            # Efficiently compute guide.p(x | context + token) for x ∈ guide.V.
            # These are individual characters that are aligned with the trie.
            p = self.guide.p_next(item.xs + token)

            children_node = T.children[node]
            for x in children_node:
                if x is None:
                    yield Item(
                        ps=item.ps * P[node] * mass[children_node[None]],
                        xs=item.xs + token,
                        ys=item.ys + (token,),
                    )

                    continue

                y = children_node[x]

                P_y = P[node] * p[x]

                if P_y > 0:
                    P[y] = P_y
                    agenda[token + x, y] = -P_y * mass[y]
