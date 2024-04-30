"""
Language model steering methods
"""
import numpy as np
import asyncio
import warnings

from arsenal.maths import sample_dict
from functools import lru_cache

from genparse.lm import LM
from genparse.cfglm import EOS
from genparse.inference import importance_sampling, smc_steer
from genparse.util import normalize
from genparse import Float

#____________________________________________________________________________________
#

class BruteForceGlobalProductOfExperts:

    def __init__(self, lm1, lm2, MAX_LENGTH):
        # Create a reference distribution for the global product of experts by
        # materializing the distrbution over strings up to a maximum length
        self.p1 = lm1.cfg.cnf.language(MAX_LENGTH).normalize()
        self.p2 = lm2.cfg.cnf.language(MAX_LENGTH).normalize()
        self.target = Float.chart({x: self.p1[x] * self.p2[x] for x in self.p1
                                   if len(x) <= MAX_LENGTH}).normalize()

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

    def __call__(self, ys):    # TODO: use log probs instead?
        assert ys[-1] == EOS
        p = 1
        for t in range(len(ys)):
            p *= self.p_next(ys[:t])[ys[t]]
        return p

    @lru_cache(None)
    def p_next(self, prefix):

        ys = tuple(prefix)
        p1 = self.lm1.p_next(ys)
        p2 = self.lm2.p_next(ys)

        # TODO: p_next should already be normalized!  Alternatively, we could
        # allow them to be energy-based models.
        p1 = normalize(p1)
        p2 = normalize(p2)

        # Below, we could alternatively use p2's support; any `k` that's not in
        # both must have probability zero.
        return normalize({k: (p1[k] * p2[k]) for k in p1})


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

            p1 = normalize(p1)
            p2 = normalize(p2)

            # Some people call this the "locally optimal proposal distribution,"
            # What does it optimize?
            q = Float.chart({k: (p1[k] * p2[k]) for k, v in p1.items()})

            Z = q.sum()

            q = normalize(q)

            if len(ys) > MAX_LENGTH:
                #print("FORCED EOS")i
                warnings.warn('force </s>')
                y = EOS

            else:
                y = sample_dict(q)

            #self.weight += np.log(p1[y] * p2[y] / (q[y] / Z))
            #self.weight += np.log(p1[y]) + np.log(p2[y]) - np.log(q[y]) + np.log(Z)
            #self.weight += np.log(Z)

            self.weight += np.log(Z)
            self.Q += np.log(q[y])

            #self.weight += np.log(p1(token | history) / p2(prev_token | prev_history))

            #self.weight += np.log(p1[y] * p2[y] / (q[y] / Z))
#            if q[y] > 0 and p1[y] > 0 and p2[y] > 0:
#                self.weight += np.log(Z)
#            else:
#                self.weight = -np.inf

#            if q[y] > 0:
#                #self.Q += np.log(p1[y]) + np.log(p2[y]) - np.log(Z)
#                self.Q += np.log(q[y])
#            else:
#                self.Q = -np.inf

            self.ys.append(y)

        def __repr__(self):
            return repr(self.ys)

    if METHOD == 'is':
        return asyncio.run(importance_sampling(Particle(), n_particles=n_particles))
    elif METHOD == 'smc':
        return asyncio.run(smc_steer(Particle(), n_particles=n_particles, n_beam=1))
    else:
        raise ValueError(METHOD)
