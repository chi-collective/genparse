from genparse import CFG


# TODO: replace this code with the transduction version!
class PrefixGrammar(CFG):
    """
    Left-derivative transformation returns a grammar that computes all left
    derivatives when it is intersected with a straight-line automaton accepting
    a given input string.
    """

    def __init__(self, parent):
        self.parent = parent
        other = self._other
        free = self._free
        top = self._top
        super().__init__(
            S=top(parent.S),
            V=parent.V,
            R=parent.R,
        )

        # Our construction for `other` assumes that there are new empty strings
        # to get those back we add one more kind of item that unions them.
        #
        # TODO: can we merge `top` with `other`?
        for x in parent.N:
            self.add(self.R.one, top(x), free(x))
            self.add(self.R.one, top(x), other(x))

        # keep all of the original rules
        for r in parent:
            self.add(r.w, r.head, *r.body)

        # invisible suffix.  These are empty "future strings".  The rules add
        # 'free' rules with the exact same structure, but different base cases,
        # as they generate empty strings only
        for x in parent.V:
            self.add(self.R.one, free(x))  # generates the empty string
        for r in parent:
            self.add(r.w, free(r.head), *(free(z) for z in r.body))

        # The `other` items (better name pending) are possibly incomplete items
        # that all nonempty prefixes of their base nonterminal's language.  Top
        # is the same, but it includes the empty string.
        #
        # visible prefix - Below, we carefully move the `other`-cursor along
        # each rule body.. The `other` are such that they have an `other`-spine
        # that separates the /visible/ prefix from the /invisible/ suffix.
        for x in parent.V:
            self.add(self.R.one, other(x), x)  # generates the usual string
        for r in parent:
            for k in range(len(r.body)):
                self.add(
                    r.w,
                    other(r.head),
                    *r.body[:k],
                    other(r.body[k]),
                    *(free(z) for z in r.body[k + 1 :]),
                )

    def spawn(self, *, R=None, S=None, V=None):  # override or else we will spawn
        return CFG(
            R=self.R if R is None else R,
            S=self.S if S is None else S,
            V=set(self.V) if V is None else V,
        )

    def _other(self, x):
        return self.parent.gensym(f'{x}⚡')

    def _free(self, x):
        return self.parent.gensym(f'{x}🔥')

    def _top(self, x):
        return self.parent.gensym(f'#{x}')
