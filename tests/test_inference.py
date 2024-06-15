import numpy as np
import pandas as pd
from arsenal import colors

from genparse import CFG, CFGLM, Float
from genparse.steer import BruteForceGlobalProductOfExperts, run

# NOTE: if the MAX_LENGTH is not long enough we will see truncation bias.  An
# alternative approach is to truncate the actual distributions.  (That is
# efficient for CFG, but I am not sure it is efficient for LLMs.)
MAX_LENGTH = 10
N_PARTICLES = 5_000


def run_test(lm1, lm2):

    ref = CheckParticles(lm1, lm2, MAX_LENGTH)

    ref.check(
        run(
            lm1,
            lm2,
            MAX_LENGTH=MAX_LENGTH,
            n_particles=N_PARTICLES,
            METHOD="is",
        )
    )

    ref.check(
        run(
            lm1,
            lm2,
            MAX_LENGTH=MAX_LENGTH,
            n_particles=N_PARTICLES,
            METHOD="smc-standard",
        )
    )

    ref.check(
        run(
            lm1,
            lm2,
            MAX_LENGTH=MAX_LENGTH,
            n_particles=N_PARTICLES,
            METHOD="smc-steer",
        )
    )


# This class computes a target distribution for testing purposes and then to run
# some diagnostics to characterize the quality of the approximation.
class CheckParticles(BruteForceGlobalProductOfExperts):

    def check(self, particles):
        n_particles = len(particles)

        # TODO: weight finalization should be part of the inference algorithm!
        w = Float.chart()
        for p in particles:
            ys = tuple(p.ys)
            numerator = self.lm1(ys) * self.lm2(ys)  # use the finalized numerator!
            if numerator > 0:
                w[ys] += numerator * np.exp(-p.Q)
        empirical = w.normalize()

        df = []
        for x in self.target | empirical:
            if empirical[x] == 0 and self.target[x] == 0:
                continue
            df.append(dict(x=x, target=self.target[x], empirical=empirical[x]))

        df = pd.DataFrame(df).sort_values("target", ascending=False)
        df["rel_error"] = abs(df.target - df.empirical) / abs(df.target)
        df["rel_error"] = df.rel_error.map(highlight)

        print(df)

        print("total variation:", abs(df.target - df.empirical).sum() / 2)

        return df


def highlight(x):
    if x > 0.1:
        return colors.light.red % x
    elif x > 0.05:
        return colors.yellow % x
    else:
        return colors.green % x


def test_empty():
    # this pair of PCFGs have no strings in common other than the empty string.
    # However, when we sample left to right it always looks like we could
    # complete the string from lm2 under lm1's palindrome constraints - so we
    # will generate forever!

    run_test(
        CFGLM.from_string(
            """

        0.45: S -> a S a
        0.45: S -> b S b
        0.1: S ->

        """
        ),
        CFGLM.from_string(
            """

        0.5: S -> a b S
        0.5: S ->

        """
        ),
    )


def test_finite_finite():

    run_test(
        CFGLM.from_string(
            """

        1: S -> a a a
        1: S -> b b b
        1: S -> b b b b b b b b b
        1: S ->

        """
        ),
        CFGLM.from_string(
            """

        2: S -> a a a
        1: S -> b b b b b
        1: S -> b b b b b b b b b

        """
        ),
    )


def test_palindrome_universal():

    run_test(
        CFGLM.from_string(
            """

        0.45: S -> a S a
        0.45: S -> b S b
        0.1: S ->

        """
        ),
        CFGLM.from_string(
            """

        0.8: S -> a S
        0.1: S -> b S
        0.1: S ->

        """
        ),
    )


def test_palindrome_finite():

    run_test(
        CFGLM.from_string(
            """

        0.45: S -> a S a
        0.45: S -> b S b
        0.1: S ->

        """
        ),
        CFGLM.from_string(
            """

        1: S -> a a a a a a a a
        1: S -> a a a a a a
        1: S -> a a a a
        1: S -> a a
        1: S ->

        """
        ),
    )


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
