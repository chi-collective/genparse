"""
Language model steering methods
"""

import asyncio
import warnings
import numpy as np
from arsenal.maths import logsumexp, sample_dict, log_sample
from arsenal import colors
from copy import deepcopy

from hfppl import Model

from genparse.inference import (
    importance_sampling,
    smc_standard,
    smc_standard_record,
    smc_steer,
)
from genparse.semiring import Float
from genparse.util import set_seed


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

    def reweight(self, weight):
        new = deepcopy(self)
        new.weight = weight
        return new

    async def step(self):
        (token, _, weight_update) = await self.proposal.sample_next_token(
            prompt=self.prompt, context=tuple(self.context)
        )
        self.context.append(token)
        self.weight += np.log(weight_update)
        self.max_tokens -= 1

        if self.verbosity > 1:
            print(f"`{token}` : {''.join(self.context)} : {self.weight}")

        if token == self.llm.eos or self.max_tokens == 0 or token == self.guide.eos:
            self.finish()
            return

    def immutable_properties(self):
        return ['llm', 'prompt', 'guide', 'proposal', 'verbosity']

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f'{self.weight:.2f}:\t'
            + colors.light.cyan % '['
            + (colors.light.cyan % '|').join(
                # [colors.bg.magenta, '%s'][i % 2] % repr(y)[1:-1] for i, y in enumerate(self.context)
                repr(y)[1:-1]
                for y in self.context
            )
            + colors.light.cyan % ']'
        )

    def __lt__(self, other):
        return self.weight < other.weight


class HFPPLSampler:
    def __init__(self, llm, guide):
        """
        Args:
            llm (TokenizedLLM)
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

        # tokenize the user's prompt if it was provided as a string rather than
        # a token sequence (tuple)
        if not isinstance(prompt, tuple):
            # TODO: consider token healing by default
            prompt = tuple(self.llm._decode[i] for i in self.llm.tokenizer.encode(prompt))

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
        self.size = len(particles)
        self.log_weights = np.array([p.weight for p in self.particles])
        self.log_total = logsumexp(self.log_weights)

        # log-marginal likelihood estimate (Note: need to exponentiate to have
        # an unbiased estimate of the true marginal likelihood).
        self.log_ml = self.log_total - np.log(self.size)

        # log-normalized weights
        self.log_normalized_weights = self.log_weights - self.log_total

        # Compute the effective sample size
        self.log_ess = -logsumexp(2 * self.log_normalized_weights)
        self.ess = np.exp(self.log_ess)

        self.record = record
        posterior = Float.chart()
        for p, w in zip(self.particles, self.log_normalized_weights):
            posterior[''.join(p.context)] += np.exp(w)
        self.posterior = posterior.normalize().sort_descending()

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

    def finalize(self, eos):
        "Optionally, we can zero-out invalid particles, i.e., those that do not end in eos."
        particles = deepcopy(self.particles)
        for p in particles:
            if p.context[-1] != eos:
                p.weight = float('-inf')
        return ParticleApproximation(particles, self.record)

    def show(self):
        for p in sorted(self, reverse=True):
            print(p)

    def resample(self):
        indices = log_sample(self.log_normalized_weights, size=self.size)
        avg_weight = self.log_total - np.log(self.size)
        return ParticleApproximation(
            [self.particles[i].reweight(avg_weight) for i in indices]
        )
