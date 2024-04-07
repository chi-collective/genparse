import pandas as pd
import numpy as np
from collections import Counter
from arsenal import colors
from arsenal.maths import compare

from genparse.steer import LocalProduct, run, normalize
from genparse.cfglm import Real, CFG, CFGLM, add_EOS, explode


MAX_LENGTH = 10
N_PARTICLES = 20_000
#METHOD = 'smc'
METHOD = 'is'


def run_test(lm1, lm2):

    Q = LocalProduct(lm1, lm2)

    particles = run(
        lm1,
        lm2,
        MAX_LENGTH = MAX_LENGTH,
        n_particles = N_PARTICLES,
        METHOD = METHOD
    )
    n_particles = len(particles)

    w = Counter()
#    q = Counter()
    for p in particles:
        x = tuple(p.ys)
#        empirical[x] += lm1.cfg(x) * lm2.cfg(x) / np.exp(p.weight) / n_particles
#        empirical[x] += lm1.cfg(x) * lm2.cfg(x) / np.exp(p.Q) / n_particles
#        empirical[x] += np.exp(p.P - p.Q) / n_particles
        w[x] += np.exp(p.weight) / n_particles
#        q[x] += np.exp(p.Q) #/ n_particles

    #print(colors.line(80))
    #q = {tuple(p.ys): np.exp(p.Q) for p in particles}
    #Q = {x: Q(x) for x in q}
    #print(colors.line(80))
    #compare(q, Q).show()

#    have = {}
#    want = {}
#    for p in particles:
#        x = tuple(p.ys)
#        have[x] = np.exp(p.P)
#        want[x] = lm1.cfg(x) * lm2.cfg(x)
#    compare(have, want).show()

#    print(w)
#    print(q)

    empirical = normalize(w)

    p1 = normalize(lm1.cfg.language(MAX_LENGTH))
    p2 = normalize(lm2.cfg.language(MAX_LENGTH))

    target = normalize(Counter({x: p1[x] * p2[x] for x in p1 | p2}))

    df = []
    for x in p1 | p2 | empirical:
        if empirical[x] == 0 and target[x] == 0: continue
        df.append(dict(x=x, target=target[x], empirical=empirical[x]))

    df = pd.DataFrame(df).sort_values('target', ascending=False)
    #f = np.log
    f = lambda x: x
    df['rel_error'] = abs(f(df.target) - f(df.empirical)) / abs(f(df.target))
    #df['ratio'] = df.target / df.empirical

    def highlight(val):
        if val > 0.1:     return colors.light.red % val
        elif val > 0.05:  return colors.yellow % val
        else:             return colors.green % val

    df['rel_error'] = df.rel_error.map(highlight)

    print(df)

    compare(target, empirical)#.show()


def test_empty():
    # this pair of PCFGs have no strings in common.  However, when we sample
    # left to right it always looks like we could complete the string from lm2
    # under lm1's palindrome constraints - so we will generate forever!

    run_test(

        CFGLM(add_EOS(CFG.from_string("""

        0.45: S -> a S a
        0.45: S -> b S b
        0.1: S ->

        """, Real))),

        CFGLM(add_EOS(CFG.from_string("""

        0.5: S -> a b S
        0.5: S ->

        """, Real))),

    )


def test_finite_finite():

    run_test(

        CFGLM(add_EOS(CFG.from_string("""

        1: S -> a a a
        1: S -> b b b
        1: S -> b b b b b b b b b
        1: S ->

        """, Real))),

        CFGLM(add_EOS(CFG.from_string("""

        2: S -> a a a
        1: S -> b b b b b
        1: S -> b b b b b b b b b

        """, Real))),
    )


def test_palindrome_universal():

    run_test(

        CFGLM(add_EOS(CFG.from_string("""

        0.45: S -> a S a
        0.45: S -> b S b
        0.1: S ->

        """, Real))),

        CFGLM(add_EOS(CFG.from_string("""

        0.8: S -> a S
        0.1: S -> b S
        0.1: S ->

        """, Real))),
    )


def test_palindrome_finite():

    run_test(

        CFGLM(add_EOS(CFG.from_string("""

        0.45: S -> a S a
        0.45: S -> b S b
        0.1: S ->

        """, Real))),

        CFGLM(add_EOS(CFG.from_string("""

        1: S -> a a a a a a a a
        1: S -> a a a a a a
        1: S -> a a a a
        1: S -> a a
        1: S ->

        """, Real))),
    )


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
