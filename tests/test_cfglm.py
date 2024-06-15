"""
Fast computation of the posterior distrubtion over the next word in a PCFG language model.
"""

from arsenal import colors

import genparse
import genparse.examples
from genparse import CFG, CFGLM, Chart, Float, Real
from genparse.cfglm import add_EOS


def fast_posterior(cfg, prefix):
    return CFGLM(cfg).p_next(tuple(prefix))


def next_token_weights_slow(cfg, prefix):
    """
    Compute the posterior over the next word in O(V N³) time.
    """
    Z = cfg.prefix_weight(prefix)
    p = cfg.R.chart()
    for v in sorted(cfg.V):
        p[v] = cfg.prefix_weight([*prefix, v])
        if p[v] == cfg.R.zero:
            del p[v]
    return Z, p


# def next_token_weights_less_slow(cfg, prefix):
#    """
#    Compute the posterior over the next word in O(V N²) time.
#    """
#    N = len(prefix)
#    P = cfg.prefix_grammar.cnf
#    chart = P._parse_chart(prefix)
#    Z = chart[0, P.S, N]
#    p = {}
#    for v in sorted(cfg.V):
#        p[v] = extend_chart(P, chart, [*prefix, v])[0, P.S, N+1]
#        if p[v] == cfg.R.zero: del p[v]
#    return Z, p


def test_new_abcdx():

    cfg = add_EOS(
        CFG.from_string(
            """

    1: S -> a b c d
    1: S -> a b c x
    1: S -> a b x x
    1: S -> a x x x
    1: S -> x x x x

    """,
            Real,
        )
    )

    for prefix in ["", "a", "ab", "abc", "abcd", "acbde"]:
        print()
        print(colors.light.blue % prefix)
        want = next_token_weights_slow(cfg, prefix)[1]
        print(want)
        have = fast_posterior(cfg, prefix)
        print(have)
        err = have.metric(want)
        print(colors.mark(err <= 1e-5))
        assert err <= 1e-5, err


def test_new_palindrome():

    cfg = add_EOS(genparse.examples.palindrome_ab)

    for prefix in ["", "a", "ab"]:
        print()
        print(colors.light.blue % prefix)
        want = next_token_weights_slow(cfg, prefix)[1]
        print(want)
        have = fast_posterior(cfg, prefix)
        print(have)
        err = have.metric(want)
        print(colors.mark(err <= 1e-5))
        assert err <= 1e-5


def test_new_papa():

    cfg = add_EOS(genparse.examples.papa)

    for prefix in [
        [],
        ["papa"],
        ["papa", "ate"],
        ["papa", "ate", "the"],
        ["papa", "ate", "the", "caviar"],
    ]:
        print()
        print(colors.light.blue % prefix)
        want = next_token_weights_slow(cfg, prefix)[1]
        print(want)
        have = fast_posterior(cfg, prefix)
        print(have)
        print(colors.mark(have.metric(want) <= 1e-5))
        assert have.metric(want) <= 1e-5


def test_sample():
    cfg = CFGLM(genparse.examples.papa)
    sample = cfg.sample(prob=True)
    print(sample)


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
