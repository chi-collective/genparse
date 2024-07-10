"""
Fast computation of the posterior distrubtion over the next word in a PCFG language model.
"""

import pytest
from arsenal import colors

import examples
from genparse import CFG, Float, add_EOS
from genparse.parse.cky import CKYLM


def fast_posterior(cfg, prefix):
    return CKYLM(cfg).p_next(tuple(prefix))


def next_token_weights_slow(cfg, prefix):
    """
    Compute the posterior over the next word in O(V N³) time.
    """
    p = cfg.R.chart()
    for v in sorted(cfg.V):
        p[v] = cfg.prefix_weight([*prefix, v])
        if p[v] == cfg.R.zero:
            del p[v]
    return p


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
            Float,
        )
    )

    for prefix in ['', 'a', 'ab', 'abc', 'abcd', 'dcba']:
        print()
        print(colors.light.blue % prefix)
        want = next_token_weights_slow(cfg, prefix).normalize()
        print(want)
        have = fast_posterior(cfg, prefix)
        print(have)
        err = have.metric(want)
        print(colors.mark(err <= 1e-5))
        assert err <= 1e-5, err

    prefix = 'acbde'
    print()
    print(colors.light.blue % prefix)
    # with pytest.raises(AssertionError):
    #    next_token_weights_slow(cfg, prefix).normalize()
    with pytest.raises(AssertionError):
        fast_posterior(cfg, prefix)


def test_new_palindrome():
    cfg = add_EOS(examples.palindrome_ab)

    for prefix in ['', 'a', 'ab']:
        print()
        print(colors.light.blue % prefix)
        want = next_token_weights_slow(cfg, prefix).normalize()
        print(want)
        have = fast_posterior(cfg, prefix)
        print(have)
        err = have.metric(want)
        print(colors.mark(err <= 1e-5))
        assert err <= 1e-5


def test_new_papa():
    cfg = add_EOS(examples.papa)

    for prefix in [
        [],
        ['papa'],
        ['papa', 'ate'],
        ['papa', 'ate', 'the'],
        ['papa', 'ate', 'the', 'caviar'],
    ]:
        print()
        print(colors.light.blue % prefix)
        want = next_token_weights_slow(cfg, prefix).normalize()
        print(want)
        have = fast_posterior(cfg, prefix)
        print(have)
        print(colors.mark(have.metric(want) <= 1e-5))
        assert have.metric(want) <= 1e-5


def test_sample():
    cfg = CKYLM(examples.papa)
    sample = cfg.sample(prob=True)
    print(sample)


def test_lm():
    from genparse.lm import LM

    cfg = CKYLM(examples.papa)
    sample = cfg.sample(prob=False) + (cfg.eos,)
    print(sample)

    assert Float.metric(cfg(sample), LM.__call__(cfg, sample)) <= 1e-8


def test_clear_cache():
    cfg = CKYLM(examples.papa)
    assert len(cfg.model._chart) == 0
    sample = cfg.sample(prob=False) + (cfg.eos,)
    p = cfg(sample)
    assert len(cfg.model._chart) > 0
    print(p, sample)
    cfg.clear_cache()
    assert len(cfg.model._chart) == 0


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
