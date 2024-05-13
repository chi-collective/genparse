import numpy as np
import time

from genparse import CFG
from genparse.semiring import Float, MaxTimes
from genparse.experimental.earley import Earley


def test_has_unary_cycle():

    cfg = CFG.from_string("""
        0.5: S → A
        0.5: A → B
        0.5: B → C
        0.5: C → A
        0.5: C → c
    """, Float)

    assert cfg.has_unary_cycle()

    cfg = CFG.from_string("""
        0.5: S → A
        0.5: A → B
        0.5: B → C
        0.5: C → c
    """, Float)

    assert not cfg.has_unary_cycle()


def test_parse_unambiguous():
    cfg = CFG.from_string("""
        1.0: S → A B
        0.3: B → A B
        0.5: A → a
        0.4: B → b
    """, Float)

    earley = Earley(cfg)
    assert_equal(earley("ab"), 0.2)
    assert_equal(earley("aab"), 0.03)
    assert_equal(earley("aaab"), 0.0045)


def test_parse_left_recursive():

    cfg = CFG.from_string("""
        1.0: S → A B
        0.3: A → A B
        0.5: A → a
        0.4: B → b
    """, Float)

    earley = Earley(cfg)
    assert_equal(earley("ab"), 0.2)
    assert_equal(earley("abb"), 0.024)
    assert_equal(earley("abbb"), 0.00288)


def assert_equal(have, want, tol=1e-10):
    if isinstance(have, (float, int)):
        error = Float.metric(have, want)
    else:
        error = have.metric(want)
    assert error <= tol, f'have = {have}, want = {want}, error = {error}'


def test_parse_unary():
    # grammar contains non-cyclic unary rules
    cfg = CFG.from_string("""
        1.0: S → B
        0.3: B → A B
        0.2: B → A
        0.5: A → a
    """, Float)

    earley = Earley(cfg)
    assert_equal(earley("a"), 0.1)
    assert_equal(earley("aa"), 0.015)
    assert_equal(earley("aaa"), 0.00225)

    cfg = CFG.from_string("""
        1.0: S → A
        0.5: S → c A
        0.3: A → B
        0.2: B → C
        0.5: C → c
    """, Float)

    earley = Earley(cfg)
    assert_equal(earley("c"), 0.03)
    assert_equal(earley("cc"), 0.015)


def test_parse_mixed():

    cfg = CFG.from_string("""
        1.0: S → a B c D
        0.4: S → A b
        0.1: B → b b
        0.5: A → a
        0.3: D → d
    """, Float)

    earley = Earley(cfg)
    assert_equal(earley("ab"), 0.2)
    assert_equal(earley("abbcd"), 0.03)


def test_parse_ambiguous_real():

    cfg = CFG.from_string("""
        1.0: S → A
        0.4: A → A + A
        0.1: A → A - A
        0.5: A → a
    """, Float)

    earley = Earley(cfg)
    assert_equal(earley("a"), 0.5)
    assert_equal(earley("a+a"), 0.1)
    assert_equal(earley("a+a+a"), 0.04)

    cfg = CFG.from_string("""
        0.4: A → A + A
        0.1: A → A - A
        0.5: A → a
    """, Float, start="A")

    earley = Earley(cfg)
    assert_equal(earley("a"), 0.5)
    assert_equal(earley("a+a"), 0.1)
    assert_equal(earley("a+a+a"), 0.04)


def test_parse_ambiguous_maxtimes():

    cfg = CFG.from_string("""
        1.0: S → A
        0.4: A → A + A
        0.1: A → A - A
        0.5: A → a
    """, MaxTimes)

    earley = Earley(cfg)
    assert_equal(earley("a"), MaxTimes(0.5))
    assert_equal(earley("a+a"), MaxTimes(0.1))
    assert_equal(earley("a+a+a"), MaxTimes(0.02))


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
