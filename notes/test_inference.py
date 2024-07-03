import numpy as np
import pandas as pd
import asyncio
import warnings
from arsenal import colors
from arsenal.maths import sample_dict

from genparse import EarleyLM, EOS, Float
from genparse.inference import smc_standard, smc_steer, importance_sampling

# NOTE: if the MAX_LENGTH is not long enough we will see truncation bias.  An
# alternative approach is to truncate the actual distributions.  (That is
# efficient for CFG, but I am not sure it is efficient for LLMs.)
MAX_LENGTH = 10
N_PARTICLES = 5_000


class BruteForceGlobalProductOfExperts:
    def __init__(self, lm1, lm2, MAX_LENGTH):
        # Create a reference distribution for the global product of experts by
        # materializing the distrbution over strings up to a maximum length
        self.lm1 = lm1
        self.lm2 = lm2
        self.p1 = lm1.cfg.cnf.materialize(MAX_LENGTH).normalize()
        self.p2 = lm2.cfg.cnf.materialize(MAX_LENGTH).normalize()
        self.target = (self.p1 * self.p2).normalize()


# TODO: We could create a class for this specific proposal distribution, which
# assumes that the lm1 and lm2 (which we'd normally call llm and guide) are
# aligned in their predictions, i.e., they have the same alphabets.
def run(lm1, lm2, *, MAX_LENGTH, n_particles, METHOD):
    # This interface is used in HFPPL / LLamPPL
    class Particle:
        def __init__(self, ys=None):
            self.ys = ys
            self.weight = 0.0

            self.Q = 0.0

        def start(self):
            self.ys = []

        def done_stepping(self):
            return EOS in self.ys

        def untwist(self):  # unused
            pass

        async def step(self):
            ys = tuple(self.ys)

            p1 = lm1.p_next(ys)
            p2 = lm2.p_next(ys)

            # TODO: p_next should already be normalized!  Skipping the
            # normalization below would allow energy-based models.
            p1 = p1.normalize()
            p2 = p2.normalize()

            # assert np.allclose(p1.sum(), 1), p1.sum()
            # assert np.allclose(p2.sum(), 1), p2.sum()

            q_ = p1 * p2

            Z = q_.sum()

            q = q_.normalize()

            if len(ys) > MAX_LENGTH:
                warnings.warn('force </s>')
                y = EOS

            else:
                y = sample_dict(q)

            # self.weight += np.log(p1[y] * p2[y] / (q[y] / Z))
            # self.weight += np.log(p1[y]) + np.log(p2[y]) - np.log(q[y]) + np.log(Z)
            # self.weight += np.log(Z)

            self.weight += np.log(Z)
            self.Q += np.log(q[y]) if q[y] > 0 else -np.inf

            # self.weight += np.log(p1(token | history) / p2(prev_token | prev_history))

            self.ys.append(y)

        def __repr__(self):
            return repr(self.ys)

    if METHOD == 'is':
        return asyncio.run(importance_sampling(Particle(), n_particles=n_particles))
    elif METHOD == 'smc-steer':
        return asyncio.run(smc_steer(Particle(), n_particles=n_particles, n_beam=1))
    elif METHOD == 'smc-standard':
        return asyncio.run(smc_standard(Particle(), n_particles=n_particles))
    else:
        raise AssertionError(METHOD)


def run_test(lm1, lm2):
    ref = CheckParticles(lm1, lm2, MAX_LENGTH)

    ref.check(
        run(
            lm1,
            lm2,
            MAX_LENGTH=MAX_LENGTH,
            n_particles=N_PARTICLES,
            METHOD='is',
        )
    )

    ref.check(
        run(
            lm1,
            lm2,
            MAX_LENGTH=MAX_LENGTH,
            n_particles=N_PARTICLES,
            METHOD='smc-standard',
        )
    )

    ref.check(
        run(
            lm1,
            lm2,
            MAX_LENGTH=MAX_LENGTH,
            n_particles=N_PARTICLES,
            METHOD='smc-steer',
        )
    )


# This class computes a target distribution for testing purposes and then to run
# some diagnostics to characterize the quality of the approximation.
class CheckParticles(BruteForceGlobalProductOfExperts):
    def check(self, particles):
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

        df = pd.DataFrame(df).sort_values('target', ascending=False)
        df.loc[:, 'rel_error'] = (abs(df.target - df.empirical) / abs(df.target)).map(
            highlight
        )

        print(df)

        print('total variation:', abs(df.target - df.empirical).sum() / 2)

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
        EarleyLM.from_string(
            """
            0.45: S -> a S a
            0.45: S -> b S b
            0.1: S ->
            """
        ),
        EarleyLM.from_string(
            """
            0.5: S -> a b S
            0.5: S ->
            """
        ),
    )


def test_finite_finite():
    run_test(
        EarleyLM.from_string(
            """
            1: S -> a a a
            1: S -> b b b
            1: S -> b b b b b b b b b
            1: S ->
            """
        ),
        EarleyLM.from_string(
            """
            2: S -> a a a
            1: S -> b b b b b
            1: S -> b b b b b b b b b
            """
        ),
    )


def test_palindrome_universal():
    run_test(
        EarleyLM.from_string(
            """
            0.45: S -> a S a
            0.45: S -> b S b
            0.1: S ->
            """
        ),
        EarleyLM.from_string(
            """
            0.8: S -> a S
            0.1: S -> b S
            0.1: S ->
            """
        ),
    )


def test_palindrome_finite():
    run_test(
        EarleyLM.from_string(
            """
            0.45: S -> a S a
            0.45: S -> b S b
            0.1: S ->
            """
        ),
        EarleyLM.from_string(
            """
            1: S -> a a a a a a a a
            1: S -> a a a a a a
            1: S -> a a a a
            1: S -> a a
            1: S ->
            """
        ),
    )


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
