from collections import namedtuple
from arsenal import colors
import numpy as np
from arsenal.timer import Benchmark
from arsenal.maths import logsumexp, log_sample, sample_dict
import atexit
import warnings
from genparse import Float


class BatchStepper:
    def __init__(self, batch_proposal, batch_llm, max_tokens):
        self.batch_proposal = batch_proposal
        self.batch_llm = batch_llm
        self.eos = batch_proposal.eos
        self.max_tokens = max_tokens
        self.timer = Benchmark('VLLM vs CFG+trie+weight')

        print(
            f'Initialized batch stepper with eos={self.eos} and max_tokens={self.max_tokens}'
        )

        atexit.register(self.cleanup)

    def batch_next_token_probs(self, particles, is_initial):
        return self.batch_llm.execute_request(particles=particles, is_initial=is_initial)

    def batch_particle_extensions(self, particles, logprobs, particle_id_to_logprob_id):
        return self.batch_proposal.execute_request(
            particles=particles,
            logprobs=logprobs,
            particle_id_to_logprob_id=particle_id_to_logprob_id,
        )

    def batch_step(self, particles, is_initial=False):
        with self.timer['vllm']:
            logprobs, particle_id_to_logprob_id = self.batch_next_token_probs(
                particles, is_initial
            )

        assert all(
            p.done
            for i, p in enumerate(particles)
            if i not in particle_id_to_logprob_id.keys()
        ), 'There are uncompleted particles which do not have a logprob'

        with self.timer['parser']:
            extensions, extension_id_to_particle_id = self.batch_particle_extensions(
                particles, logprobs, particle_id_to_logprob_id
            )

        assert all(
            p.done
            for i, p in enumerate(particles)
            if i not in extension_id_to_particle_id.values()
        ), 'There are uncompleted particles which do not have an extension'

        for extension_id, particle_id in extension_id_to_particle_id.items():
            particle = particles[particle_id]
            extension = extensions[extension_id]
            particles[particle_id] = Particle(
                prompt=particle.prompt,
                log_weight=particle.log_weight + extension.log_weight,
                context=particle.context + (extension.token,),
                context_ids=particle.context_ids + (extension.token_id,),
                done=(
                    extension.token == self.eos
                    or len(particle.context) + 1 >= self.max_tokens
                ),
                parent=particle.parent,
            )

        return particles

    def cleanup(self):
        warnings.warn('Cleaning up batch stepper. All subprocess will be terminated.')
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


def maybe_resample(particles, ess_threshold):
    log_weights = [p.log_weight for p in particles]
    log_total = logsumexp(log_weights)

    log_normalized_weights = log_weights - log_total

    log_ess = -logsumexp(2 * log_normalized_weights)
    ess = np.exp(log_ess)

    if ess < ess_threshold:
        n_particles = len(particles)
        indices = log_sample(log_normalized_weights, size=n_particles)
        avg_log_weight = log_total - np.log(n_particles)
        return [
            particles[i]._replace(log_weight=avg_log_weight, parent=i) for i in indices
        ]
    else:
        return particles


def pretty_print_particles(particles):
    for i, p in enumerate(particles):
        print(f'├ Particle {i:3d} `{p.context[-1]}` : {p}')


def importance_sampling(batch_model, n_particles, verbosity=0):
    """Standard sequential importance sampling.

    Args:
      - `batch_model` (`BatchStepper`): The model to perform inference on.
      - `n_particles` (`int`): The number of particles to perform inference with.
      - `verbosity` (`int`): Verbosity level. When > 0, particles are printed at each step.

    Returns:
      - `particles` (`list[Particles]`): The completed particles after inference.

    TODO: Add record
    """
    particles = init_particles(n_particles)

    try:
        particles = batch_model.batch_step(particles, is_initial=True)
        if verbosity > 0:
            pretty_print_particles(particles)

        while not all(p.done for p in particles):
            particles = batch_model.batch_step(particles)
            if verbosity > 0:
                pretty_print_particles(particles)

    except Exception as e:
        batch_model.cleanup()
        raise e

    return ParticleApproximation(particles)


def smc(batch_model, n_particles, ess_threshold=0.5, verbosity=0):
    """Standard sequential Monte Carlo algorithm with multinomial resampling.

    Args:
      - `batch_model` (`BatchStepper`): The model to perform inference on.
      - `n_particles` (`int`): The number of particles to perform inference with.
      - `ess_threshold` (`float`): Effective sample size below which resampling
         triggered, given as a fraction of `n_particles`.
      - `verbosity` (`int`): Verbosity level. When > 0, particles are printed at each step.

    Returns:
      - `particles` (`list[Particles]`): The completed particles after inference.

    TODO: Add record
    """
    particles = init_particles(n_particles)

    try:
        particles = batch_model.batch_step(particles, is_initial=True)
        particles = maybe_resample(particles, n_particles * ess_threshold)
        if verbosity > 0:
            pretty_print_particles(particles)

        while not all(p.done for p in particles):
            particles = batch_model.batch_step(particles)
            particles = maybe_resample(particles, n_particles * ess_threshold)
            if verbosity > 0:
                pretty_print_particles(particles)

    except Exception as e:
        batch_model.cleanup()
        raise e

    return ParticleApproximation(particles)
