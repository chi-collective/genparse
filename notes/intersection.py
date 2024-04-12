import genparse
from genparse import CFG, Real
from itertools import product
from collections import defaultdict

from genparse.cfg import FSA


# reference implementation of the intersection algorithm
def intersect_slow(self, fsa):
    "Return a CFG that denoting the pointwise product of `self` and `fsa`."
    if isinstance(fsa, (str, list, tuple)): fsa = FSA.from_string(fsa, self.R)
    new_start = self.S
    new = self.spawn(S = new_start)
    for r in self:
        for qs in product(fsa.states, repeat=1+len(r.body)):
            new.add(r.w, (qs[0], r.head, qs[-1]), *((qs[i], r.body[i], qs[i+1]) for i in range(len(r.body))))
    for qi, wi in fsa.start.items():
        for qf, wf in fsa.stop.items():
            new.add(wi*wf, new_start, (qi, self.S, qf))
    for i, a, j, w in fsa.arcs():
        new.add(w, (i, a, j), a)
    return new


def _intersect_bottom_up(self, fsa):
    "Determine which items of the intersected grammar are supported"

    A = set()

    I = defaultdict(set)   # incomplete items
    C = defaultdict(set)   # complete items
    R = defaultdict(set)   # rules indexed by first subgoal; non-nullary

    for r in self:
        if len(r.body) > 0:
            R[r.body[0]].add(r)

    # we have two base cases:
    #
    # base case 1: arcs
    for i, a, j, w in fsa.arcs():
        A.add((i, a, (), j))

    # base case 2: nullary rules
    for r in self:
        if len(r.body) == 0:
            for i in fsa.states:
                A.add((i, r.head, (), i))

    # drain the agenda
    while A:
        (i, X, Ys, j) = A.pop()

        # No pending items ==> the item is complete
        if not Ys:

            if j in C[i, X]: continue
            C[i, X].add(j)

            # combine the newly completed item with incomplete rules that are
            # looking for an item like this one
            for (h, X1, Zs) in I[i, X]:
                A.add((h, X1, Zs[1:], j))

            # initialize rules that can start with an item like this one
            for r in R[X]:
                A.add((i, r.head, r.body[1:], j))

        # Still have pending items ==> advanced the pending items
        else:

            if (i, X, Ys) in I[j, Ys[0]]: continue
            I[j, Ys[0]].add((i, X, Ys))

            for k in C[j, Ys[0]]:
                A.add((i, X, Ys[1:], k))

    return C


def intersect_fast(self, fsa):
    "Return a CFG that denoting the pointwise product of `self` and `fsa`."
    if isinstance(fsa, (str, list, tuple)): fsa = FSA.from_string(fsa, self.R)

    new_start = self.S
    new = self.spawn(S = new_start)

    # The bottom-up intersection algorithm is a two pass algorithm
    #
    # Pass 1: Determine the set of items that are possiblly nonzero-valued
    C = _intersect_bottom_up(self, fsa)

    # Note that over estimate is safe so we could even use the set below,
    # however, it would be much less efficient to do so.
    #
#    C = {(i,X): fsa.states for i in fsa.states for X in self.N | self.V}

    # Pass 2: expands the grammar's rules against those items; Although, the
    # construction we have is correct for unbinarized rules, it is generally
    # much more efficient to binarized the grammar before calling this method.
    #
    def product(start, Ys):
        """
        Helper method; expands the rule body

        Given Ys = [Y_1, ... Y_K], we will enumerate expansion of the form

        (s_0, Y_1, s_1), (s_1, Y_2, s_2), ..., (s_{k-1}, Y_K, s_K)

        where each (s_k, Y_k, s_k) in the expansion is a completed items
        (i.e., \forall k: (s_k, Y_k, s_k) in C).
        """
        if not Ys:
            yield []
        else:
            for K in C[start, Ys[0]]:
                for rest in product(K, Ys[1:]):
                    yield [(start, Ys[0], K)] + rest

    start = {I for (I,_) in C}

    for r in self:
        if len(r.body) == 0:
            for s in fsa.states:
                new.add(r.w, (s, r.head, s))
        else:
            for I in start:
                for rhs in product(I, r.body):
                    K = rhs[-1][-1]
                    new.add(r.w, (I, r.head, K), *rhs)

    for i, wi in fsa.start.items():
        for k, wf in fsa.stop.items():
            new.add(wi*wf, new_start, (i, self.S, k))

    for i, a, j, w in fsa.arcs():
        if a in self.V:
            new.add(w, (i, a, j), a)

    return new


def test_palindrome1():
    cfg = CFG.from_string("""
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """, Real)

    fsa = FSA.from_string('aa', cfg.R)

    check(cfg, fsa)


def test_palindrome2():
    cfg = CFG.from_string("""
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """, Real)

    fsa = FSA(
        Real.chart(),
        [
            (0, 'a', 0, Real.one),
            (0, 'b', 0, Real.one),
            (0, 'c', 0, Real.one),
        ],
        Real.chart(),
    )

    fsa.start[0] = Real.one
    fsa.stop[0] = Real.one

    check(cfg, fsa)


def test_palindrome3():
    cfg = CFG.from_string("""
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """, Real)

    fsa = FSA(
        Real.chart(),
        [
            # straight line aaa
            (0, 'a', 1, Real.one),
            (1, 'a', 2, Real.one),
            (2, 'a', 3, Real.one),
            # and then a cycle
            (3, 'a', 3, Real(0.5)),
            (3, 'b', 3, Real(0.5)),
        ],
        Real.chart(),
    )

    fsa.start[0] = Real.one
    fsa.stop[3] = Real.one

    check(cfg, fsa)


def test_catalan1():
    cfg = CFG.from_string("""
    0.4: S -> S S
    0.3: S -> a
    0.3: S -> b
    """, Real)

    fsa = FSA.from_string('aa', cfg.R)

    check(cfg, fsa)


def test_catalan2():
    cfg = CFG.from_string("""
    0.4: S -> S S
    0.3: S -> a
    0.3: S -> b
    """, Real)

    fsa = FSA(
        Real.chart(),
        [
            (0, 'a', 0, Real.one),
            (0, 'b', 0, Real.one),
            (0, 'c', 0, Real.one),
        ],
        Real.chart(),
    )

    fsa.start[0] = Real.one
    fsa.stop[0] = Real.one

    check(cfg, fsa)


def test_catalan2():
    cfg = CFG.from_string("""
    0.4: S -> S S
    0.3: S -> a
    0.3: S -> b
    """, Real)

    fsa = FSA(
        Real.chart(),
        [
            # straight line aaa
            (0, 'a', 1, Real.one),
            (1, 'a', 2, Real.one),
            (2, 'a', 3, Real.one),
            # and then a cycle
            (3, 'a', 3, Real(0.5)),
            (3, 'b', 3, Real(0.5)),
        ],
        Real.chart(),
    )

    fsa.start[0] = Real.one
    fsa.stop[3] = Real.one

    check(cfg, fsa)



CHECK_RULES = True
CHECK_CHART = True


def check(cfg, fsa):

    want = intersect_slow(cfg, fsa).trim(bottomup_only=True)
    have = intersect_fast(cfg, fsa)

    if 0:
        want = want.trim().trim()
        have = have.trim().trim()

#    want = want.trim()
#    have = have.trim()

    print()
    print('have=')
    print(have)
    print()
    print('want=')
    print(want)

    print()
    print('have chart=')
    print(have.agenda())
    print()
    print('want chart=')
    print(want.agenda())

    assert have.treesum().metric(want.treesum()) < 1e-5, [have.treesum(), want.treesum()]

    if CHECK_CHART:
        have.agenda().assert_equal(want.agenda(), tol=1e-5)

    if CHECK_RULES:
        print()
        print(have)
        print()
        print(want)
        have.assert_equal(want, verbose=True)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
