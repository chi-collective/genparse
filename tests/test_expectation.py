from genparse.cfg import CFG
from genparse.semiring import Expectation

tol = 1e-8

cfg_1 = CFG.from_string(
    """
       <0.7,0.7>: S → a S  
       <0.3,0.3>: S → a 
    """,
    Expectation,  # The expectaton semiring takes < p(X), p(X) * X >
)

cfg_2 = CFG.from_string(
    """
       <0.9,1.8>: S → a S b  
       <0.1,0>: S →  
    """,
    Expectation,  # The expectaton semiring takes < p(X), p(X) * X >
)


cfg_finite = CFG.from_string(
    """
       <0.5,1.5>: S → a a a
       <0.5,0.5>: S → a 
    """,
    Expectation,
)


def test_1():
    """The expected lenght of a string in cfg_1 is E[L] =  Σ_{n=0}^{\infty} (n+1)*(0.7^n * 0.3) = 10/3
    Hint: differentiate the geometric series !"""

    want = 10 / 3
    have = cfg_1.treesum().score[1]

    assert abs(want - have) < tol


def test_2():
    """The expected lenght of a string in cfg_2 is E[L] =  Σ_{n=0}^{\infty} 2n*(0.9^n * 0.1) = 18
    Hint: differentiate the geometric series !"""

    want = 18
    have = cfg_2.treesum().score[1]

    assert abs(want - have) < tol


def test_finite():
    want = 2
    have = cfg_finite.treesum().score[1]

    assert abs(want - have) < tol


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
