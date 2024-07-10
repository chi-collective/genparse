import numpy as np

from genparse import Float, CFG
from genparse.experimental.gad import Sampler
from genparse.parse.earley import EarleyLM


def test_GAD_finite_1():
    cfg1 = CFG.from_string(
        """
        1: S -> a B
        0.5: B -> b A
        0.5: B -> a A
        1: A -> a
        """,
        Float,
    )

    cfg2 = CFG.from_string(
        """
        0.3: S -> a a a
        0.7: S -> b b b
        """,
        Float,
    )

    lm1 = EarleyLM(cfg1)
    lm2 = EarleyLM(cfg2)

    sampler = Sampler(lm1, lm2)

    for _ in range(0, 15):
        sampler.sample()

    print(sampler.root.mass)
    assert np.allclose(sampler.root.mass, 0.15)
    print('test passed!')


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
