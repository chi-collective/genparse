"""
Fast computation of the posterior distrubtion over the next word in a PCFG language model.
"""

import genparse
import genparse.examples
from genparse import CFG, Float, Real
from genparse.cfglm import CFGLM, next_token_weights
from genparse.chart import Chart

from arsenal import colors
from arsenal.maths import sample_dict
from functools import lru_cache
from collections import defaultdict


def fast_posterior(cfg, prefix):
    chart = cfg.prefix_grammar.cnf._parse_chart(prefix)
    return next_token_weights(cfg.prefix_grammar.cnf, chart, prefix)


#def test_cfglm():
#    lm = CFGLM(add_EOS(genparse.examples.papa))
#    ys = []
#    for _ in range(10):
#        q = lm.p_next(tuple(ys))
#        Q = {k: v.score for k, v in q.items()}
#        y = sample_dict(Q)
#        assert sum(Q.values()) > 0
#        print(ys)
#        print(q)
#        print(colors.yellow % colors.arrow.r, colors.green % y)
#        if y == EOS: break
#        ys.append(y)
#    return ys


def next_token_weights_slow(cfg, prefix):
    """
    Compute the posterior over the next word in O(V N³) time.
    """
    Z = cfg.prefix_weight(prefix)
    p = Chart(cfg.R)
    for v in sorted(cfg.V):
        p[v] = cfg.prefix_weight([*prefix, v])
        if p[v] == cfg.R.zero: del p[v]
    return Z, p


#def next_token_weights_less_slow(cfg, prefix):
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

    cfg = CFG.from_string("""

    1: S -> a b c d
    1: S -> a b c x
    1: S -> a b x x
    1: S -> a x x x
    1: S -> x x x x

    """, Real)

    for prefix in ['', 'a', 'ab', 'abc', 'abcd', 'acbde']:
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

    cfg = genparse.examples.palindrome_ab

    for prefix in ['', 'a', 'ab']:
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

    cfg = genparse.examples.papa

    for prefix in [
            [],
            ['papa'],
            ['papa', 'ate'],
            ['papa', 'ate', 'the'],
            ['papa', 'ate', 'the', 'caviar'],
    ]:
        print()
        print(colors.light.blue % prefix)
        want = next_token_weights_slow(cfg, prefix)[1]
        print(want)
        have = fast_posterior(cfg, prefix)
        print(have)
        print(colors.mark(have.metric(want) <= 1e-5))
        assert have.metric(want) <= 1e-5


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
