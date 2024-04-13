import pandas as pd
import numpy as np
from collections import Counter
from arsenal import colors
from arsenal.maths import compare

from genparse.steer import LocalProduct, run, normalize
from genparse.cfglm import Float, CFG, CFGLM, add_EOS, explode


MAX_LENGTH = 10
N_PARTICLES = 10_000
#METHOD = 'smc'
#METHOD = 'is'


def run_test(lm1, lm2):

    ref = CheckParticles(lm1, lm2, MAX_LENGTH)

    ref.check(run(
        lm1,
        lm2,
        MAX_LENGTH = MAX_LENGTH,
        n_particles = N_PARTICLES,
        METHOD = 'is',
    ))

    ref.check(run(
        lm1,
        lm2,
        MAX_LENGTH = MAX_LENGTH,
        n_particles = N_PARTICLES,
        METHOD = 'smc',
    ))


# This class computes a target distribution for testing purposes and then to run
# some diagnostics to characterize the quality of the approximation.
class CheckParticles:

    def __init__(self, lm1, lm2, MAX_LENGTH):
        # Create a reference distribution for the global product of experts by
        # materializing the distrbution over strings up to a maximum length
        self.p1 = normalize(lm1.cfg.cnf.language(MAX_LENGTH))
        self.p2 = normalize(lm2.cfg.cnf.language(MAX_LENGTH))
        self.target = normalize(Counter({x: self.p1[x] * self.p2[x] for x in self.p1
                                         if len(x) <= MAX_LENGTH}))

    def check(self, particles):
        n_particles = len(particles)

        w = Counter()
        for p in particles:
            w[tuple(p.ys)] += np.exp(p.weight)
        empirical = normalize(w)

        df = []
        for x in self.p1 | self.p2 | empirical:
            if empirical[x] == 0 and self.target[x] == 0: continue
            df.append(dict(x=x, target=self.target[x], empirical=empirical[x]))

        df = pd.DataFrame(df).sort_values('target', ascending=False)
        df['rel_error'] = abs(df.target - df.empirical) / abs(df.target)
        df['rel_error'] = df.rel_error.map(highlight)

        print(df)

        print('total variation:', abs(df.target - df.empirical).sum() / 2)

        #compare(target, empirical)#.show()
        return df


def highlight(x):
    if   x > 0.1:  return colors.light.red % x
    elif x > 0.05: return colors.yellow    % x
    else:          return colors.green     % x


def test_empty():
    # this pair of PCFGs have no strings in common other than the empty string.
    # However, when we sample left to right it always looks like we could
    # complete the string from lm2 under lm1's palindrome constraints - so we
    # will generate forever!

    run_test(

        CFGLM(add_EOS(CFG.from_string("""

        0.45: S -> a S a
        0.45: S -> b S b
        0.1: S ->

        """, Float))),

        CFGLM(add_EOS(CFG.from_string("""

        0.5: S -> a b S
        0.5: S ->

        """, Float))),

    )


def test_finite_finite():

    run_test(

        CFGLM(add_EOS(CFG.from_string("""

        1: S -> a a a
        1: S -> b b b
        1: S -> b b b b b b b b b
        1: S ->

        """, Float))),

        CFGLM(add_EOS(CFG.from_string("""

        2: S -> a a a
        1: S -> b b b b b
        1: S -> b b b b b b b b b

        """, Float))),
    )


def test_palindrome_universal():

    run_test(

        CFGLM(add_EOS(CFG.from_string("""

        0.45: S -> a S a
        0.45: S -> b S b
        0.1: S ->

        """, Float))),

        CFGLM(add_EOS(CFG.from_string("""

        0.8: S -> a S
        0.1: S -> b S
        0.1: S ->

        """, Float))),
    )


def test_palindrome_finite():

    run_test(

        CFGLM(add_EOS(CFG.from_string("""

        0.45: S -> a S a
        0.45: S -> b S b
        0.1: S ->

        """, Float))),

        CFGLM(add_EOS(CFG.from_string("""

        1: S -> a a a a a a a a
        1: S -> a a a a a a
        1: S -> a a a a
        1: S -> a a
        1: S ->

        """, Float))),
    )


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
