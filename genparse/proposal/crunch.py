from arsenal.datastructures.pdict import pdict
from collections import namedtuple
from arsenal.iterextras import head_iter
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

    def posterior_enumerate(self, prompt, depth):
        Q = pdict()

        it = head_iter(self._iter_p_next(Item(1, (), prompt)))
        Q[it] = -1

        while Q:
            iterator = Q.pop()

            if iterator.done:
                continue
            item = next(iterator)
            print('.', end='')

            if item.xs[-1] == self.T.new_eos:
                if self.guide(''.join(item.xs)):
                    yield item
                continue

            if len(item.ys) - len(prompt) <= depth:
                extend_iter = head_iter(self._iter_p_next(item))
                if not extend_iter.done:
                    Q[extend_iter] = -extend_iter.head.ps

            if not iterator.done:
                Q[iterator] = -iterator.head.ps

    def _iter_p_next(self, item):
        """
        This method will lazily enumerate the nodes in the intersection of `llm` and
        and the `guide` for the given context using the TokenProposal.
        """
        for token, value in self.T.traverse_trie(item.xs, self.llm.p_next(item.ys)):
            yield Item(
                item.ps * value,
                item.xs + (token,),
                item.ys + ((token,) if token != self.T.new_eos else (self.T.old_eos,)),
            )
