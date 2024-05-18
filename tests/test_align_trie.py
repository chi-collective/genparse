import numpy as np
import random
from arsenal import timeit, colors
from arsenal.maths import sample_dict
from genparse.util import LarkStuff
from genparse.inference import TraceSWOR
from genparse import CFGLM, locally_normalize, Float, EOS
from genparse.lm import GreedilyTokenizedLLM
from genparse.align.trie import TokenTrieApproximation


def test_llm_trie_approximation():
    np.random.seed(0)
    random.seed(0)

    pcfg = CFGLM(locally_normalize(LarkStuff(r"""

    start: /[ ]*Tim(othy)?[ ](Fabbri[ ])?Vieira\./

    """).char_cfg(.99), tol=1e-100))

    prompt = 'Hello my name is'
    llm = GreedilyTokenizedLLM("gpt2")

    llm.sample('My name is', verbose=1, max_tokens=10)

    token_trie_approx = TokenTrieApproximation(llm, pcfg)
    tracer = TraceSWOR()
    W = Float.chart()
    for _ in range(10):
        with tracer:
            print('----------------------------------')
            with timeit('complete sample'):
                ys, q = token_trie_approx.sample(prompt, max_tokens=50,
                                                 draw=tracer, prob=True,
                                                 verbosity=1)

            score = llm(ys) * pcfg(ys + EOS)
            W[ys] += score / q

            print(colors.light.yellow % 'sample:', ys)

            print(W.normalize())


def test_chomsky_said():
    np.random.seed(0)
    random.seed(0)

    pcfg = CFGLM(locally_normalize(LarkStuff(r"""

    start: /Noam[ ]Chomsky[ ]famously[ ]wrote,[ ]"/ expr /\."/

    expr: /[A-Za-z0-9,; ]+/
//    expr: /[Tt]ime[ ]flies[ ]like[ ]an[ ]arrow/
//        | /[iI][ ]like[ ]to[ ]dance/
//        | /Colorless[ ]green[ ]ideas[ ]sleep[ ]furiously/

    """).char_cfg(.9999), tol=1e-300))

    #print(''.join(pcfg.sample()))

#    from genparse.semiring import Log
#    tmp = pcfg.cfg.spawn(R = Log)
#    for r in pcfg.cfg:
#        tmp.add(Log(np.log(r.w)), r.head, *r.body)
#    lpcfg = CFGLM(tmp)

#    x = 'Noam Chomsky famously wrote, "One of the most outrageous things about Modernity has always been muckraking in human nature; it has deceptively distorted the way in which one views human rights by making dece'
#    lp = lpcfg.p_next(x)
#    pp = pcfg.p_next(x)
#    from IPython import embed; embed()

    prompt = ' '
    llm = GreedilyTokenizedLLM("gpt2")

    W = Float.chart()

    token_trie_approx = TokenTrieApproximation(llm, pcfg)
#    tracer = TraceSWOR()
    tracer = sample_dict
    for _ in range(10):
#        with tracer:
            print('----------------------------------')
            with timeit('complete sample'):
                ys, q = token_trie_approx.sample(prompt, max_tokens=50,
                                                 draw=tracer, prob=True,
                                                 verbosity=1)
            score = llm(ys) * pcfg(ys + EOS)
            W[ys] += score / q

            print(q)

            print(colors.light.yellow % 'sample:', ys)

            print(W.normalize())


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
