import genparse
from genparse import CFG, Float, Real
from itertools import product
from collections import defaultdict
from genparse.fst import FST
from genparse.wfsa import WFSA, EPSILON


def assert_equal(have, want, tol=1e-5):
    if isinstance(have, (int, float)):
        error = abs(have - want)
    else:
        error = have.metric(want)
    assert error <= tol, f'have = {have}, want = {want}, error = {error}'


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
        assert a != EPSILON
        new.add(w, (i, a, j), a)
    return new


# reference implementation of the composition algorithm; does not support input-side epsilon arcs
def compose_slow(self, fst):
    "Reference implementation of the grammar-transducer composition."
    if isinstance(fst, (str, list, tuple)): fst = FST.from_string(fst, self.R)
    new_start = self.S
    new = self.spawn(S = new_start)
    for r in self:
        for qs in product(fst.states, repeat=1+len(r.body)):
            new.add(r.w, (qs[0], r.head, qs[-1]), *((qs[i], r.body[i], qs[i+1]) for i in range(len(r.body))))
    for qi, wi in fst.start.items():
        for qf, wf in fst.stop.items():
            new.add(wi*wf, new_start, (qi, self.S, qf))
    for i, (a, b), j, w in fst.arcs():
        assert a != EPSILON
        if b == EPSILON :
            new.add(w, (i, a, j))
        else:
            new.add(w, (i, a, j), b)
    return new


def check_fst(cfg, fst):

    want = compose_slow(cfg, fst).trim(bottomup_only=True)
    have = cfg @ fst  # fast composition

    if 0:
        want = want.trim().trim()
        have = have.trim().trim()

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

    assert_equal(have.treesum(), want.treesum())

    print()
    print(have)
    print()
    print(want)
    have.assert_equal(want, verbose=True)


def test_palindrome1():
    cfg = CFG.from_string("""
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """, Float)

    fsa = WFSA.from_string('aa', cfg.R)

    check(cfg, fsa)


def test_palindrome2():
    cfg = CFG.from_string("""
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """, Real)

    fsa = WFSA(Real)
    fsa.add_arc(0, 'a', 0, Real.one)
    fsa.add_arc(0, 'b', 0, Real.one)
    fsa.add_arc(0, 'c', 0, Real.one)

    fsa.add_I(0, Real.one)
    fsa.add_F(0, Real.one)

    check(cfg, fsa)


def test_palindrome3():
    cfg = CFG.from_string("""
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """, Real)

    fsa = WFSA(Real)
    # straight line aaa
    fsa.add_arc(0, 'a', 1, Real.one)
    fsa.add_arc(1, 'a', 2, Real.one)
    fsa.add_arc(2, 'a', 3, Real.one)
    # and then a cycle
    fsa.add_arc(3, 'a', 3, Real(0.5))
    fsa.add_arc(3, 'b', 3, Real(0.5))

    fsa.add_I(0, Real.one)
    fsa.add_F(3, Real.one)

    check(cfg, fsa)


def test_catalan1():
    cfg = CFG.from_string("""
    0.4: S -> S S
    0.3: S -> a
    0.3: S -> b
    """, Real)

    fsa = WFSA.from_string('aa', cfg.R)

    check(cfg, fsa)


def test_catalan2():
    cfg = CFG.from_string("""
    0.4: S -> S S
    0.3: S -> a
    0.3: S -> b
    """, Real)

    fsa = WFSA(Real)
    fsa.add_I(0, Real.one)
    fsa.add_F(3, Real.one)

    # straight line aaa
    fsa.add_arc(0, 'a', 1, Real.one)
    fsa.add_arc(1, 'a', 2, Real.one)
    fsa.add_arc(2, 'a', 3, Real.one)
    # and then a cycle
    fsa.add_arc(3, 'a', 3, Real(0.5))
    fsa.add_arc(3, 'b', 3, Real(0.5))

    check(cfg, fsa)


def check(cfg, fsa):

    want = intersect_slow(cfg, fsa).trim(bottomup_only=True)
    have = cfg @ fsa # fast intersection

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

    assert_equal(have.treesum(), want.treesum())

    print()
    print(have)
    print()
    print(want)
    have.assert_equal(want, verbose=True)


# COMPOSITION TESTS

def test_catalan_fst():
    cfg = CFG.from_string("""
    0.4: S -> S S
    0.3: S -> a
    0.3: S -> b
    """, Real)

    fst = FST(Real)

    fst.add_I(0, Real(1.0))
    fst.add_arc(0, ('a', 'b'), 1, Real(1.0))
    fst.add_arc(1, ('a', 'b'), 2, Real(1.0))
    fst.add_arc(2, ('a', EPSILON ), 3, Real(1.0))
    fst.add_arc(3, ('a', 'b'), 3, Real(1.0))
    fst.add_arc(3, ('b', 'a'), 3, Real(1.0))
    fst.add_F(3, Real(1.0))

    check_fst(cfg, fst)


def test_palindrome_fst():
    cfg = CFG.from_string("""
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """, Real)

    fst = FST(Real)

    fst.add_I(0, Real(1.0))
    fst.add_arc(0, ('a', 'b'), 1, Real(1.0))
    fst.add_arc(1, ('a', 'b'), 2, Real(1.0))
    fst.add_arc(2, ('a', 'b'), 3, Real(1.0))
    fst.add_arc(3, ('a', EPSILON ), 3, Real(1.0))
    fst.add_arc(3, ('b', EPSILON ), 3, Real(1.0))
    fst.add_F(3, Real(1.0))

    check_fst(cfg, fst)


# TEST FOR COMPOSITION WITH EPSILON INPUT ARCS

def test_epsilon_fst():
    cfg = CFG.from_string("""
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """, Real)

    fst = FST(Real)

    fst.add_I(0, Real(1.0))
    fst.add_arc(0, ('a', 'a'), 1, Real(1.0))
    fst.add_arc(1, (EPSILON , 'a'), 2, Real(1.0))
    fst.add_arc(2, ('a', 'a'), 3, Real(1.0))
    fst.add_arc(3, (EPSILON, 'b'), 4, Real(1.0))
    fst.add_F(4, Real(1.0))

    fst_removed = FST(Real)

    fst_removed.add_I(0, Real(1.0))
    fst_removed.add_arc(0, ('a','a'),1, Real(1.0))
    fst_removed.add_arc(1, ('a','a'),2, Real(1.0))
    fst_removed.add_F(2, Real(1.0))

    have = cfg.compose_epsilon_fast(fst) 
    want = cfg @ fst_removed

    assert_equal(want.treesum(), have.treesum() )

    #trim check
    have.assert_equal(have.trim(bottomup_only = True))

def test_epsilon_fst_2():
    #This test case is a bit more complex as it contains epsilon cycles on the FST
    cfg = CFG.from_string("""
    0.3: S -> a S a
    0.4: S -> b S b
    0.3: S ->
    """, Real)

    fst = FST(Real)

    fst.add_I(0, Real(1.0))
    fst.add_arc(0, ('a','a'), 1, Real(1.0))
    fst.add_arc(1, (EPSILON, EPSILON), 1, Real(0.5))
    fst.add_arc(1, ('a','a'), 2, Real(1.0))
    fst.add_F(2, Real(1.0))
    
    fst_removed = FST(Real)

    fst_removed.add_I(0, Real(1.0))
    fst_removed.add_arc(0, ('a','a'), 1, Real(2.0)) # The weight of the cycle has been pushed here
    fst_removed.add_arc(1, ('a','a'), 2, Real(1.0))
    fst_removed.add_F(2, Real(1.0))

    want = cfg @ fst_removed
    have = cfg.compose_epsilon_fast(fst)

    assert_equal(want.treesum(), have.treesum())


   

    have.assert_equal(have.trim(bottomup_only = True))

if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
