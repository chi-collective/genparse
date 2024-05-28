import numpy as np
import random
from arsenal import timeit, colors
from time import time

from genparse.util import LarkStuff
from genparse import CFGLM, locally_normalize, Float
from genparse.proposal import TokenProposal, CharacterProposal
from genparse.experimental.earley import EarleyLM
from genparse.lm import make_mock_llm

from example_grammars import arith, iql_small


def test_token_arith():

    np.random.seed(0)
    random.seed(0)

    cfg = locally_normalize(LarkStuff(arith, cnf=False).char_cfg(.9), tol=1e-100).trim()

    guide = EarleyLM(cfg)

    llm = make_mock_llm()

    proposal = TokenProposal(guide=guide, llm=llm)

    samples = []
    for _ in range(10):
        samples.append(proposal.sample(prompt='', max_tokens=100, verbosity=1))
        print(samples[-1])


def test_character_arith():

    np.random.seed(0)
    random.seed(0)

    cfg = locally_normalize(LarkStuff(arith, cnf=False).char_cfg(.99), tol=1e-100).trim()

    guide = EarleyLM(cfg)

    llm = make_mock_llm()

    proposal = CharacterProposal(llm=llm, guide=guide)

    samples = []
    for _ in range(10):
        samples.append(proposal.sample(prompt='', max_tokens=100, verbosity=1))
        print(samples[-1])


def test_token_iql_small():

    np.random.seed(0)
    random.seed(0)

    cfg = locally_normalize(LarkStuff(iql_small, cnf=False).char_cfg(.99), tol=1e-100).trim()

    llm = make_mock_llm()

    guide = EarleyLM(cfg)

    proposal = TokenProposal(llm=llm, guide=guide)

    samples = []
    for _ in range(10):
        samples.append(proposal.sample(chunked=True, max_tokens=100))
        print(samples[-1])


def test_character_iql_small():

    np.random.seed(0)
    random.seed(0)

    llm = make_mock_llm()

    cfg = locally_normalize(LarkStuff(iql_small, cnf=False).char_cfg(.99), tol=1e-100).trim()

    guide = EarleyLM(cfg)

    proposal = CharacterProposal(llm=llm, guide=guide)

    samples = []
    for _ in range(10):
        samples.append(proposal.sample(prompt='', max_tokens=100, verbosity=1))
        print(samples[-1])


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
