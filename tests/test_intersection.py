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


intersect_fast = CFG.intersect


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
