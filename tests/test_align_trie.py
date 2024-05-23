import numpy as np
import random
from arsenal import timeit, colors
from arsenal.maths import sample_dict
from genparse.util import LarkStuff
from genparse import CFGLM, locally_normalize, Float, EOS
from genparse.lm import GreedilyTokenizedLLM
from genparse.align.trie import TokenTrieApproximation
from genparse import Boolean
from genparse.cfglm import BoolMaskCFGLM


def test_llm_trie_approximation():
    np.random.seed(0)
    random.seed(0)

    pcfg = CFGLM(locally_normalize(LarkStuff(r"""

    start: /[ ]*Tim(othy)?[ ](Fabbri[ ])?Vieira\./

    """).char_cfg(.99), tol=1e-100))

    prompt = 'Hello my name is'
    llm = GreedilyTokenizedLLM("gpt2")

    llm.sample('My name is', verbose=1, max_tokens=10)

    proposal = TokenTrieApproximation(llm, pcfg)
    W = Float.chart()
    for _ in range(10):
        print('----------------------------------')
        with timeit('sample'):
            ys, q = proposal.sample(prompt, max_tokens=50, prob=True, verbosity=1)

        score = llm(ys) * pcfg(ys + EOS)
        print('weight:', score, '/', q)
        W[ys] += score / q

        print(colors.light.yellow % 'sample:', ys)

        print(W.normalize())


def test_chomsky_said():
    np.random.seed(0)
    random.seed(0)

    pcfg = CFGLM(locally_normalize(LarkStuff(r"""

    start: /Noam[ ]Chomsky[ ]famously[ ]wrote,[ ]"/ expr /\."/

//    expr: /[A-Za-z0-9,; ]+/
    expr: /[Tt]ime[ ]flies[ ]like[ ]an[ ]arrow/
        | /[iI][ ]like[ ]to[ ]dance/
        | /[cC]olorless[ ]green[ ]ideas[ ]sleep[ ]furiously/

    """).char_cfg(.9999), tol=1e-300))

    # TODO: we can improve this model considerably by encoding the max length
    # into the CFG as it will push backward the constraint that `."` needs to be
    # generated.  Currently, most of the samples generated have weight zero
    # because of the '."' technicality!  Encoding the constraint exactly might
    # be a bit of a challenge as the set of tokens of length <= T is an
    # expensive constraint to encoded exactly.  But, we can approximate with
    # something simpler like the number of white spaces in many settings.

    # XXX: we are using the boolean CFG instead of the PCFG
    pcfg = BoolMaskCFGLM(pcfg.cfg)

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

    proposal = TokenTrieApproximation(llm, pcfg)
    for _ in range(10):
        print('----------------------------------')
        with timeit('sample'):
            ys, q = proposal.sample(prompt, max_tokens=100, prob=True, verbosity=1)
        score = llm(ys) * pcfg(ys + EOS)
        print('weight:', score, '/', q, '=', score / q)
        W[ys] += score / q

        print(colors.light.yellow % 'sample:', ys)

        print(W.normalize())


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
