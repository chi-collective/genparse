"""
Language model steering methods
"""

import asyncio
import warnings

import numpy as np
from arsenal.maths import logsumexp, sample_dict

from hfppl import Model

from genparse import EOS
from genparse.inference import (
    importance_sampling,
    smc_standard,
    smc_standard_record,
    smc_steer,
)
from genparse.semiring import Float
from genparse.util import set_seed


# ____________________________________________________________________________________
# Approximate inference with HFPPL
# This code is still experimental and actively being developed
# TODO: write tests


class HFPPLParticle(Model):
    """
    Simple HFPPL model (particle).
    TODO: Create a proposal interface to make this class reusable.
        I think this class should be essentially hidden to the user and only be used by the HFPPLSampler.
    """

    def __init__(self, llm, guide, proposal, prompt, max_tokens, verbosity=0):
        super().__init__()
        self.llm = llm
        self.guide = guide
        self.prompt = prompt
        self.context = []
        self.proposal = proposal
        self.max_tokens = max_tokens
        self.verbosity = verbosity

    async def step(self):
        (token, _, weight_update) = await self.proposal.sample_next_token(
            prompt=self.prompt, context=''.join(self.context)
        )
        self.context.append(token)
        self.weight += np.log(weight_update)
        self.max_tokens -= 1

        if self.verbosity > 1:
            print(f"`{token}` : {''.join(self.context)} : {self.weight}")

        if token == self.llm.eos or self.max_tokens == 0 or token == EOS:
            self.finish()
            return

    def immutable_properties(self):
        return ['llm', 'prompt', 'guide', 'verbosity']

    def __repr__(self):
        return f"`{'' if not self.context else self.context[-1]}` : {''.join(self.context)} : {self.weight}"

    def __str__(self):
        return ''.join(self.context)


class HFPPLSampler:
    def __init__(self, llm, guide):
        """
        Args:
            llm (AsyncGreedilyTokenizedLLM)
            guide (LM)
        """
        self.llm = llm
        self.guide = guide

    def run_inference(
        self,
        prompt,
        proposal,
        method,
        n_particles,
        n_beam=None,
        max_tokens=float('inf'),
        verbosity=0,
        return_record=False,
        seed=None,
    ):
        """
        Returns:
            particle_approximation (ParticleApproximation)
        """
        if seed is not None:
            set_seed(seed)

        model = HFPPLParticle(
            llm=self.llm,
            guide=self.guide,
            prompt=prompt,
            proposal=proposal,
            max_tokens=max_tokens,
            verbosity=verbosity,
        )

        record = None
        if method == 'smc-steer':
            assert n_beam is not None
            if return_record:
                warnings.warn('Record not yet implemented for smc-steer')
            particles = asyncio.run(
                smc_steer(model, n_particles=n_particles, n_beam=n_beam)
            )

        elif method == 'smc-standard':
            if n_beam is not None:
                warnings.warn('`n_beam` is set, but will be ignored by smc-standard')
            if return_record:
                particles, record = asyncio.run(
                    smc_standard_record(
                        model, n_particles=n_particles, return_record=return_record
                    )
                )
            else:
                particles = asyncio.run(smc_standard(model, n_particles=n_particles))

        elif method == 'importance-sampling':
            particles = asyncio.run(importance_sampling(model, n_particles=n_particles))

        else:
            raise ValueError(f'Unknown inference method: {method}.')

        self.llm.clear_cache()
        self.guide.clear_cache()

        return ParticleApproximation(particles, record=record)


class ParticleApproximation:
    def __init__(self, particles, record=None):
        self.particles = particles
        self.log_weights = [p.weight for p in self.particles]
        self.log_ml = logsumexp(self.log_weights) - np.log(len(self.log_weights))
        self.record = record

        posterior = Float.chart()
        for p in self.particles:
            posterior[''.join(p.context)] += np.exp(p.weight)
        self.posterior = posterior.normalize()

    def __iter__(self):
        return iter(self.particles)

    def sample(self, n=None, draw=sample_dict):
        if n is None:
            return draw(self.posterior)
        else:
            return [draw(self.posterior) for _ in range(n)]

    def __str__(self):
        return str(self.posterior)

    def _repr_html_(self):
        return self.posterior._repr_html_()

    def risk(self, kernel, candidate):
        return sum(p * kernel(candidate, y) for y, p in self.posterior.items())
