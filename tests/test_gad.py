import html
import numpy as np
from arsenal import Integerizer, colors
from arsenal.maths import sample

from graphviz import Digraph
from genparse import Float, EOS, add_EOS, CFG
from genparse.experimental.gad import Sampler, Node
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

    for i in range(0, 15):
        sampler.sample()

    print(sampler.root.mass)
    assert sampler.root.mass == 0.15
    print('test passed!')


test_GAD_finite_1()
