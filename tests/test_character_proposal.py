import numpy as np
import random
from arsenal import timeit, colors
from arsenal.maths import sample_dict
from genparse.util import LarkStuff
from genparse import CFGLM, locally_normalize, Float, Boolean, EOS
from genparse.lm import GreedilyTokenizedLLM
from genparse.proposal import CharacterProposal
from genparse.cfglm import BoolMaskCFGLM


def test_timothy():
    np.random.seed(0)
    random.seed(0)

    pcfg = CFGLM(locally_normalize(LarkStuff(r"""

    start: /[ ]*Tim(othy)?[ ](Fabbri[ ])?Vieira\./

    """).char_cfg(.99), tol=1e-100))

    prompt = 'Hello my name is'
    llm = GreedilyTokenizedLLM("gpt2")

    llm.sample('My name is', verbose=1, max_tokens=10)

    proposal = CharacterProposal(llm=llm, guide=pcfg)
    W = Float.chart()
    for _ in range(10):
        print('----------------------------------')
        with timeit('sample'):
            ys, q = proposal.sample(prompt, verbosity=1, max_tokens=50)

        score = llm(ys) * pcfg(ys + EOS)
        print('weight:', score, '/', q)
        W[ys] += score / q

        print(colors.light.yellow % 'sample:', ys)

        print(W.normalize())


def todo_chomsky():
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

    # XXX: we are using the boolean CFG instead of the PCFG; the PCFG is running
    # into numerical underflow.  We need to use the log-semiring or a rescaling
    # trick in the Earley parser.
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

    proposal = CharacterProposal(llm=llm, guide=pcfg)
    for _ in range(10):
        print('----------------------------------')
        with timeit('sample'):
            ys, q = proposal.sample(prompt, verbosity=1)
        score = llm(ys) * pcfg(ys + EOS)
        print('weight:', score, '/', q, '=', score / q)
        W[ys] += score / q

        print(colors.light.yellow % 'sample:', ys)

        print(W.normalize())


def todo_fruit():
    np.random.seed(0)
    random.seed(0)

    pcfg = CFGLM(locally_normalize(LarkStuff(r"""
    start: (|" ") sentence

    sentence: noun verb noun
            | noun verb "like" noun

    noun: det adj? NOUN
    verb: VERB
    adj: ADJ
    det: "a" | "the"

    NOUN: "flies" | "banana" | "fruit"
    VERB: "like" | "flies"
    ADJ: "smelly"

    """).char_cfg(.99, ignore='[ ]?'), tol=1e-100))

    pcfg = BoolMaskCFGLM(pcfg.cfg)


    prompt = 'The following is a favorite sentence among linguists:'

    llm = GreedilyTokenizedLLM("gpt2")

    W = Float.chart()

    proposal = CharacterProposal(llm=llm, guide=pcfg)
    for _ in range(10):
        print('----------------------------------')
        with timeit('sample'):
            ys, q = proposal.sample(prompt, verbosity=1)
        score = llm(ys) * pcfg(ys + EOS)
        print('weight:', score, '/', q, '=', score / q)
        W[ys] += score / q

        print(colors.light.yellow % 'sample:', ys)

        print(W.normalize())


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
