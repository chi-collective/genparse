"""
Language model steering methods
"""
import numpy as np
import asyncio
import warnings
from arsenal.maths import sample_dict

from genparse.lm import LM
from genparse.cfglm import EOS
from genparse.inference import importance_sampling, smc_standard, smc_steer, TraceSWOR
from genparse.util import normalize, format_table
from genparse import Float

#____________________________________________________________________________________
#

class BruteForceGlobalProductOfExperts:

    def __init__(self, lm1, lm2, MAX_LENGTH):
        # Create a reference distribution for the global product of experts by
        # materializing the distrbution over strings up to a maximum length
        self.lm1 = lm1
        self.lm2 = lm2
        self.p1 = lm1.cfg.cnf.language(MAX_LENGTH).filter(lambda x: len(x) <= MAX_LENGTH).normalize()
        self.p2 = lm2.cfg.cnf.language(MAX_LENGTH).filter(lambda x: len(x) <= MAX_LENGTH).normalize()
        self.target = (self.p1 * self.p2).normalize()


# TODO: support early termination options
class generation_tree:

    def __init__(self, lm, **opts):
        tracer = TraceSWOR()
        D = Float.chart()
        while tracer.root.mass > 0:
            with tracer:
                s, p = lm.sample(draw=tracer, prob=True, **opts)
                D[s] += p
        D = Float.chart((k, D[k]) for k in sorted(D))
        self.D = D
        self.tracer = tracer

    def _repr_html_(self):
        return format_table([[self.D, self.tracer]])


#____________________________________________________________________________________
#

class LocalProduct(LM):
    """This class implements a *local* product of experts, an LM that is derived by
    multiplying the conditional distributions of each token in a pair of
    token-synchronized LM.

    Typically, `LocalProduct` is a baseline method or a proposal distribution
    for the *global* product of experts.

    [Some people call LocalProduct the "locally optimal proposal distribution" -
    what does it actually optimize?]

    """

    def __init__(self, lm1, lm2):
        self.lm1 = lm1
        self.lm2 = lm2
        assert lm1.V == lm2.V
        assert lm1.eos == lm2.eos
        super().__init__(V = lm1.V, eos = lm1.eos)

    def __call__(self, ys):    # TODO: use log probs instead?
        assert ys[-1] == self.eos
        p = 1
        for t in range(len(ys)):
            p *= self.p_next(ys[:t])[ys[t]]
        return p

    def p_next(self, prefix):

        ys = tuple(prefix)
        p1 = self.lm1.p_next(ys)
        p2 = self.lm2.p_next(ys)

        # TODO: p_next should already be normalized!  Skipping the normalization
        # below would allow energy-based models.
        p1 = normalize(p1)
        p2 = normalize(p2)

        # Below, we could alternatively use p2's support; any `k` that's not in
        # both must have probability zero.
        return (p1 * p2).normalize()


#_______________________________________________________________________________
# Approximate inference

def run(lm1, lm2, *, MAX_LENGTH, n_particles, METHOD):

    # TODO: I'd like to have a target--proposal pair passed in and for the SMC
    # stuff to combine it in the right way.  If we pass an unnormalized target
    # (i.e., an energy), then we get a consistent semantics (i.e., we are just
    # off by the normalization constant everywhere).


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

        def untwist(self):   # unused
            pass

        async def step(self):

            ys = tuple(self.ys)

            p1 = lm1.p_next(ys)
            p2 = lm2.p_next(ys)

            # TODO: p_next should already be normalized!  Skipping the
            # normalization below would allow energy-based models.
            p1 = p1.normalize()
            p2 = p2.normalize()

            q = p1 * p2

            Z = q.sum()

            q = normalize(q)

            if len(ys) > MAX_LENGTH:
                warnings.warn('force </s>')
                y = EOS

            else:
                y = sample_dict(q)

            #self.weight += np.log(p1[y] * p2[y] / (q[y] / Z))
            #self.weight += np.log(p1[y]) + np.log(p2[y]) - np.log(q[y]) + np.log(Z)
            #self.weight += np.log(Z)

            self.weight += np.log(Z)
            self.Q += np.log(q[y]) if q[y] > 0 else -np.inf

            #self.weight += np.log(p1(token | history) / p2(prev_token | prev_history))

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
