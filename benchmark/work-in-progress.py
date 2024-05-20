from genparse.util import LarkStuff, hf_tokenizer
from genparse import CFGLM, locally_normalize, Float, add_EOS, EOS
from genparse.align import CharAlignedCFGLM
from genparse.align.trie import TokenTrieApproximation
from genparse.experimental.earley import Earley
from genparse.inference import TraceSWOR
from arsenal import timeit, colors
from time import time

from example_grammars import arith, iql_small


def test_basic_aligned_model_arithmetic():

    with timeit('grammar setup'):
        cfg = add_EOS(locally_normalize(LarkStuff(arith).char_cfg(.9), tol=1e-100).trim())

    with timeit('cfglm setup'):
        # the base character-level CFG language model
#        lm = CFGLM(cfg)
        lm = Earley(cfg.prefix_grammar.nullaryremove().unarycycleremove().renumber())

    with timeit('tokenizer setup'):
        H = hf_tokenizer()

    with timeit('CharAlignedCFGLM setup'):
        proposal = CharAlignedCFGLM(lm=lm, words={x for _, x in H.pairs}, eos=H.tokenizer.eos_token)

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


def test_graft_align_iql_small():

    with timeit('lark conversion'):
        cfg = add_EOS(locally_normalize(LarkStuff(iql_small).char_cfg(.99), tol=1e-100).trim())

    with timeit('tokenizer setup'):
        H = hf_tokenizer()

    with timeit('CFGLM setup'):
        lm1 = CFGLM(cfg)

    with timeit('Earley setup'):
        lm = Earley(cfg.prefix_grammar.nullaryremove().unarycycleremove().renumber())

    with timeit('CharAlignedCFGLM setup'):
        proposal = CharAlignedCFGLM(lm=lm, words={x for _, x in H.pairs},
                                  eos=H.tokenizer.eos_token)

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

    tracer = TraceSWOR()
    samples = []
    for _ in range(10):
        with tracer:
            samples.append(proposal.sample(draw=tracer, chunked=True))
            print(samples[-1])


def test_trie_align_iql_small():

    with timeit('lark conversion'):
        cfg = add_EOS(locally_normalize(LarkStuff(iql_small).char_cfg(.99), tol=1e-100).trim())

    with timeit('tokenizer setup'):
        H = hf_tokenizer()

    with timeit('CFGLM setup'):
        lm1 = CFGLM(cfg)

    with timeit('Earley setup'):
        guide = Earley(cfg.prefix_grammar.nullaryremove().unarycycleremove().renumber())

    h = len(H.pairs)
    u = Float.chart({w: 1/h for _, w in H.pairs})
    class MockLLM:
        eos = H.tokenizer.eos_token
        V = set(u)
        def p_next(self, context): return u


    mock_llm = MockLLM()
    with timeit('TokenTrieApproximation setup'):
        proposal = TokenTrieApproximation(mock_llm, guide)

    tracer = TraceSWOR()
    samples = []
    for _ in range(10):
        with tracer:
            samples.append(proposal.sample(prompt='', max_tokens=1000, verbosity=1, draw=tracer))
            print(samples[-1])


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
