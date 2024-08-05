from collections import namedtuple
from arsenal import colors
import numpy as np
from arsenal.maths import logsumexp, log_sample, sample_dict
import atexit
import warnings
from genparse import Float
from genparse.record import SMCRecord


class BatchStepModel:
    def __init__(self, batch_proposal, batch_llm, max_tokens, prompt=None):
        self.batch_proposal = batch_proposal
        self.batch_llm = batch_llm
        self.eos = batch_proposal.eos
        self.max_tokens = max_tokens

        if prompt is not None:
            self.set_prompt(prompt)

        atexit.register(self.cleanup)

    def set_prompt(self, prompt):
        self.batch_llm.set_prompt(prompt)

    def batch_step(self, particles, is_initial=False):
        logprobs, particle_id_to_logprob_id = self.batch_llm.batch_next_token_logprobs(
            particles=particles, is_initial=is_initial
        )

        extensions, extension_id_to_particle_id = (
            self.batch_proposal.batch_particle_extensions(
                particles=particles,
                logprobs=logprobs,
                particle_id_to_logprob_id=particle_id_to_logprob_id,
            )
        )

        for extension_id, particle_id in extension_id_to_particle_id.items():
            particle = particles[particle_id]
            extension = extensions[extension_id]
            particles[particle_id] = Particle(
                prompt=particle.prompt,
                log_weight=particle.log_weight + extension.log_weight,
                context=particle.context + (extension.token,),
                context_ids=particle.context_ids + (extension.token_id,),
                done=(
                    extension.token == self.batch_proposal.eos
                    or extension.token == self.batch_llm.eos
                    or len(particle.context) + 1 >= self.max_tokens
                ),
                parent=particle.parent,
            )

        return particles

    def cleanup(self, warn=False):
        if warn:
            warnings.warn(
                'Cleaning up batch step model. All subprocesses will be terminated.'
            )
        self.batch_proposal.cleanup()
        self.batch_llm.cleanup()


###############
# SMC methods #
###############


class Particle(
    namedtuple('Particle', 'prompt, log_weight, context, context_ids, parent, done')
):
    def __repr__(self):
        return (
            f'{self.log_weight:.2f}:\t'
            + colors.light.cyan % '['
            + (colors.light.cyan % '|').join(repr(y)[1:-1] for y in self.context)
            + colors.light.cyan % ']'
        )


class ParticleApproximation:
    def __init__(self, particles, record=None):
        self.particles = list(particles)
        self.size = len(particles)
        self.log_weights = np.array([p.log_weight for p in self.particles])
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
                p if p.context[-1] == eos else p._replace(log_weight=float('-inf'))
                for p in self
            ],
            self.record,
        )

    def show(self):
        for p in sorted(self, reverse=True):
            print(p)


def init_particles(n_particles):
    return [Particle((), 0, (), (), None, False) for _ in range(n_particles)]


def pretty_print_particles(particles, step_info):
    for i, p in enumerate(particles):
        print(f'├ Particle {i:3d} `{p.context[-1]}` : {p}')
    print(
        f"│ Step {step_info['step']:3d} average weight: {step_info['average_weight']:.4f}"
    )


def maybe_resample(particles, ess_threshold, return_record, step_info, verbosity):
    if return_record:
        step_info['particles'] = [
            {'context': p.context, 'weight': p.log_weight} for p in particles
        ]

    n_particles = len(particles)
    log_weights = [p.log_weight for p in particles]
    log_total = logsumexp(log_weights)
    avg_log_weight = log_total - np.log(n_particles)

    if return_record or verbosity > 0:
        step_info['average_weight'] = avg_log_weight

    log_normalized_weights = log_weights - log_total
    log_ess = -logsumexp(2 * log_normalized_weights)
    ess = np.exp(log_ess)

    if ess < n_particles * ess_threshold:
        indices = log_sample(log_normalized_weights, size=n_particles)
        particles = [
            particles[i]._replace(log_weight=avg_log_weight, parent=i) for i in indices
        ]

        if return_record:
            step_info['resample_indices'] = indices

        if verbosity > 0:
            print(
                f'└╼  Resampling! {indices}. Weights all set to = {avg_log_weight:.4f}.'
            )
    else:
        if verbosity > 0:
            print('└╼')

    return particles


def smc(batch_model, n_particles, ess_threshold=0.5, verbosity=0, return_record=False):
    """Standard sequential Monte Carlo algorithm with multinomial resampling.

    Args:
      - `batch_model` (`BatchStepper`): The model to perform inference on.
      - `n_particles` (`int`): The number of particles to perform inference with.
      - `ess_threshold` (`float`): Effective sample size below which resampling
         triggered, given as a fraction of `n_particles`.
      - `verbosity` (`int`): Verbosity level. When > 0, particles are printed at each step.

    Returns:
      - `particle_approximation` (`ParticleApproximation`): The completed particle approximation.
    """

    record = (
        SMCRecord(
            {
                'n_particles': n_particles,
                'ess_threshold': ess_threshold,
                'algorithm': 'smc_standard_record',
                'history': [],
            }
        )
        if return_record
        else None
    )

    particles = init_particles(n_particles)
    step_num = 1
    while not all(p.done for p in particles):
        step_info = {'step': step_num}

        particles = batch_model.batch_step(particles, is_initial=step_num == 1)
        particles = maybe_resample(
            particles, ess_threshold, return_record, step_info, verbosity
        )

        if verbosity > 0:
            pretty_print_particles(particles, step_info)

        if return_record:
            record['history'].append(step_info)

        step_num += 1

    return ParticleApproximation(particles, record)


def importance_sampling(batch_model, n_particles, verbosity=0, return_record=False):
    """Standard sequential importance sampling.

    Args:
      - `batch_model` (`BatchStepper`): The model to perform inference on.
      - `n_particles` (`int`): The number of particles to perform inference with.
      - `verbosity` (`int`): Verbosity level. When > 0, particles are printed at each step.

    Returns:
      - `particle_approximation` (`ParticleApproximation`): The completed particle approximation.
    """
    return smc(
        batch_model=batch_model,
        n_particles=n_particles,
        ess_threshold=0,
        verbosity=verbosity,
        return_record=return_record,
    )
