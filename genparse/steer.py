"""
Language model steering methods
"""

import asyncio
import numpy as np

from arsenal import colors
from arsenal.maths import logsumexp, sample_dict, log_sample
from copy import deepcopy
from collections import namedtuple

from genparse.record import SMCRecord
from genparse.semiring import Float
from genparse.util import set_seed


class Particle(namedtuple('Particle', 'weight, context, parent, done')):
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


class Sampler:
    def __init__(self, llm, guide):
        """
        Args:
            llm (TokenizedLLM)
            guide (LM)
        """
        self.llm = llm
        self.guide = guide

    async def step(self, p):
        if p.done:
            return p

        (token, _, log_weight_update) = await self.proposal.sample_next_token(
            prompt=self.prompt,
            context=p.context,
        )

        new_particle = Particle(
            weight=p.weight + log_weight_update,
            context=p.context + (token,),
            parent=p,
            done=(
                token == self.llm.eos
                or token == self.guide.eos
                or 1 + len(p.context) == self.max_tokens
            ),
        )

        if self.verbosity > 1:
            print(new_particle)

        return new_particle

    def run_inference(
        self,
        prompt,
        proposal,
        method,
        n_particles,
        max_tokens=float('inf'),
        ess_threshold=0.5,
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

        # TODO: this should be in the constructor; maybe proposal should just
        # have smc methods on them.
        self.proposal = proposal
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.verbosity = verbosity

        particles = ParticleApproximation(
            [
                Particle(weight=0, context=(), parent=None, done=False)
                for _ in range(n_particles)
            ]
        )

        record = None

        if method == 'smc-standard':
            particles, record = asyncio.run(
                smc_standard(
                    particles,
                    step=self.step,
                    return_record=return_record,
                    verbosity=verbosity,
                    ess_threshold=ess_threshold,
                )
            )

        elif method == 'importance-sampling':
            particles, record = asyncio.run(
                smc_standard(
                    particles,
                    step=self.step,
                    return_record=return_record,
                    verbosity=verbosity,
                    ess_threshold=0,
                )
            )

        else:
            raise ValueError(f'Unknown inference method: {method}.')

        self.llm.clear_cache()
        self.guide.clear_cache()

        return ParticleApproximation(particles, record=record)


class ParticleApproximation:
    def __init__(self, particles, record=None):
        self.particles = list(particles)
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

    @property
    def posterior(self):
        posterior = Float.chart()
        for p, prob in zip(self.particles, np.exp(self.log_normalized_weights)):
            posterior[''.join(p.context)] += prob
        return posterior.normalize().sort_descending()

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.particles)

    def __getitem__(self, i):
        return self.particles[i]

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
        return ParticleApproximation(
            [
                p if p.context[-1] == eos else p._replace(weight=float('-inf'))
                for p in self
            ],
            self.record,
        )

    def show(self):
        for p in sorted(self, reverse=True):
            print(p)

    def resample(self):
        indices = log_sample(self.log_normalized_weights, size=self.size)
        avg_weight = self.log_total - np.log(self.size)
        return ParticleApproximation(
            [self.particles[i]._replace(weight=avg_weight) for i in indices]
        )


async def smc_standard(
    particles, step, ess_threshold=0.5, return_record=True, verbosity=0
):
    """Standard sequential Monte Carlo algorithm with multinomial resampling.

    Modified version `smc_standard` that keeps a record of information about the
    run.  Should be identical in behavior, but uses a different format to store
    the record.

    Args:
      - `particles` (`ParticleApproximation`): The model to perform inference on.
      - `ess_threshold` (`float`): Effective sample size below which resampling
         triggered, given as a fraction of `particles.size`.

    Returns:
      - `particles` (`list[hfppl.modeling.Model]`): The completed particles after inference.
      - `record` (`SMCRecord`): Information about inference run history.

    """

    # Initialize record dict
    record = (
        SMCRecord(
            {
                'n_particles': particles.size,
                'ess_threshold': ess_threshold,
                'algorithm': 'smc_standard_record',
                'history': [],
            }
        )
        if return_record
        else None
    )

    if return_record or verbosity > 0:
        step_num = 1

    while not all(p.done for p in particles):
        if return_record:
            step_info = {'step': step_num}

        particles = ParticleApproximation(
            await asyncio.gather(*[step(p) for p in particles])
        )

        # Compute log average weight (used if resampling, else only for record)
        avg_weight = particles.log_total - np.log(particles.size)
        if verbosity > 0:
            for i, p in enumerate(particles):
                print(
                    f'├ Particle {i:3d} (weight {p.weight:.4f}). `{p.context[-1]}` : {p}'
                )
            print(f'│ Step {step_num:3d} average weight: {avg_weight:.4f}')

        if return_record:
            step_info['particles'] = [
                {'context': p.context, 'weight': p.weight} for p in particles
            ]
            step_info['average_weight'] = avg_weight

        # Resample if necessary
        if particles.ess < ess_threshold * particles.size:
            # resampling: sample indices to copy
            resampled_indices = log_sample(
                particles.log_normalized_weights, size=particles.size
            )

            # resampled_indices.sort()  # removed. sorting should be done in post if necessary
            particles = ParticleApproximation(
                [
                    particles[i]._replace(weight=avg_weight, parent=particles[i])
                    for i in resampled_indices
                ]
            )

            if return_record:
                step_info['resample_indices'] = resampled_indices

            if verbosity > 0:
                print(
                    f'└╼  Resampling! {resampled_indices}. Weights all set to = {avg_weight:.4f}.'
                )
        else:
            if verbosity > 0:
                print('└╼')

        if return_record or verbosity > 0:
            step_num += 1
            if return_record:
                record['history'].append(step_info)

    return particles, record
