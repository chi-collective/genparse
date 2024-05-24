import numpy as np
import random
from arsenal import timeit, colors
from time import time

from genparse.util import LarkStuff
from genparse import CFGLM, locally_normalize, Float, add_EOS, EOS
from genparse.proposal import TokenProposal, CharacterProposal
from genparse.experimental.earley import Earley
from genparse.lm import make_mock_llm

from example_grammars import arith, iql_small


def test_graft_arith():

    np.random.seed(0)
    random.seed(0)

    cfg = add_EOS(locally_normalize(LarkStuff(arith, cnf=False).char_cfg(.9), tol=1e-100).trim())

    lm = Earley(cfg.prefix_grammar.nullaryremove().unarycycleremove().renumber())

    mock_llm = make_mock_llm()

    proposal = TokenProposal(lm=lm, words=mock_llm.V, eos=mock_llm.eos)

    #===========================================================================
    # [2024-04-29 Mon] OPTIMIZATIONS: Tianyu provided the follow contexts, which
    # include a mix of slow and not so slow strings
    #
    #===========================================================================
    contexts = [
        #'Bey',
        #'BeyAk',
        'BeyAk=-',
        #'BeyAk=-amide',
        #'BeyAk=-amide*',
        #'BeyAk=-amide*pread',
        #'BeyAk=-amide*pread+(',
        #'BeyAk=-amide*pread+(Dust',
        #'BeyAk=-amide*pread+(Dusts',
        #'BeyAk=-amide*pread+(Dusts)--',
    ]

    for context in contexts:
        print(colors.yellow % repr(context))
        with timeit('parsing'):
            b4 = time()
            p = proposal.p_next(context)
            took = time() - b4
        print('output size:', len(p))
        print('time/output:', took / len(p))


def test_trie_arith():

    np.random.seed(0)
    random.seed(0)

    cfg = add_EOS(locally_normalize(LarkStuff(arith, cnf=False).char_cfg(.99), tol=1e-100).trim())

    guide = Earley(cfg.prefix_grammar.nullaryremove().unarycycleremove().renumber())

    mock_llm = make_mock_llm()

    proposal = CharacterProposal(mock_llm, guide)

    samples = []
    for _ in range(10):
        samples.append(proposal.sample(prompt='', max_tokens=1000, verbosity=1))
        print(samples[-1])


def test_graft_iql_small():

    np.random.seed(0)
    random.seed(0)

    cfg = add_EOS(locally_normalize(LarkStuff(iql_small, cnf=False).char_cfg(.99), tol=1e-100).trim())

    mock_llm = make_mock_llm()

    lm = Earley(cfg.prefix_grammar.nullaryremove().unarycycleremove().renumber())

    proposal = TokenProposal(lm=lm, words=mock_llm.V, eos=mock_llm.eos)

    #===============================================================================
    # [2024-04-28 Sun] CKY optimizations:
    # initial version (best of several runs)  16.7135 sec
    # using integers for nonterminals:        14.4915 sec;  1.15x faster
    # using lists of charts to avoid copying: 13.0670 sec;  1.1x faster
    # left-child loop in `extend_chart`)       6.0260 sec;  2.2x faster
    # left-child loop in `next_token_weights`  1.6176 sec;  3.7x faster
    #===============================================================================

    with timeit('took'):

        p = proposal.p_next('')
        print(p)
        assert p.keys() == {'S', 'SE', 'SELECT'}

        p = proposal.p_next('SELECT * FROM data')
        print(p)
        assert p.keys() == {' ', ' <', ' </', ' G', ' W', ' O', ' GR', ' WH', ' OR',
                            ' GROUP', ' WHERE', ' ORDER'}

        p = proposal.p_next('SELECT age FROM data')
        print(p)
        assert p.keys() == {' ', ' <', ' </', ' G', ' W', ' O', ' GR', ' WH', ' OR',
                            ' GROUP', ' WHERE', ' ORDER'}

    samples = []
    for _ in range(10):
        samples.append(proposal.sample(chunked=True))
        print(samples[-1])


def test_trie_iql_small():

    np.random.seed(0)
    random.seed(0)

    mock_llm = make_mock_llm()

    cfg = add_EOS(locally_normalize(LarkStuff(iql_small, cnf=False).char_cfg(.99), tol=1e-100).trim())

    guide = Earley(cfg.prefix_grammar.nullaryremove().unarycycleremove().renumber())

    proposal = CharacterProposal(mock_llm, guide)

    samples = []
    for _ in range(10):
        samples.append(proposal.sample(prompt='', max_tokens=1000, verbosity=1))
        print(samples[-1])


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
