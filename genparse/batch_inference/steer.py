import copy
import atexit
import warnings
import numpy as np
from arsenal import colors
from genparse import Float
from numpy.random import random
from genparse.record import SMCRecord
from arsenal.maths import logsumexp, log_sample, sample_dict


class BatchStepModel:
    """
    A model for performing batch steps during inference.

    Attributes:
        batch_proposal (BatchProposal): The batch proposal.
        batch_llm (BatchLLM): The batch language model.
        max_tokens (int): The maximum number of tokens to sample.

    """

    def __init__(self, batch_proposal, batch_llm, max_tokens, prompt=None):
        self.batch_proposal = batch_proposal
        self.batch_llm = batch_llm
        self.max_tokens = max_tokens

        if prompt is not None:
            self.set_prompt(prompt)

        atexit.register(self.cleanup)

    def set_prompt(self, prompt):
        """
        Set the prompt for the batch language model.

        Args:
            prompt (str): The prompt to set.
        """
        self.batch_llm.set_prompt(prompt)

    def batch_step(self, particles, is_initial=False):
        """
        Perform a batch step during inference.
        Computes the next token logprobs for each particle.
        Samples an extension for each particle and updates the particle weights.

        Args:
            particles (list): List of Particle objects.
            is_initial (bool, optional): Flag indicating if it is the initial step. Defaults to False.

        Returns:
            list: List of updated Particle objects after the batch step.
        """
        logprobs_by_seq_group, particle_idx_to_logprob_idx = (
            self.batch_llm.batch_next_token_logprobs(
                particles=particles, is_initial=is_initial
            )
        )

        extensions, extension_idx_to_particle_idx = (
            self.batch_proposal.batch_particle_extensions(
                particles=particles,
                logprobs_by_seq_group=logprobs_by_seq_group,
                particle_idx_to_logprob_idx=particle_idx_to_logprob_idx,
            )
        )

        for extension_idx, extension in enumerate(extensions):
            particle_idx = extension_idx_to_particle_idx[extension_idx]
            particle = particles[particle_idx]
            particles[particle_idx] = Particle(
                prompt=particle.prompt,
                log_weight=particle.log_weight + extension.log_weight,
                log_potential=particle.log_potential,
                context=particle.context + (extension.token,),
                context_ids=particle.context_ids + (extension.token_id,),
                done=(
                    extension.token == self.batch_proposal.eos
                    or extension.token_id == self.batch_llm.eos_token_id
                    or len(particle.context) + 1 >= self.max_tokens
                ),
                parent=particle.parent,
            )

        return particles

    def cleanup(self):
        """
        Clean up resources used by the batch proposal and batch language model.
        """
        self.batch_proposal.cleanup()
        self.batch_llm.cleanup()


########################
# Inference algorithms #
########################


class Particle:
    def __init__(
        self,
        prompt=None,
        log_weight=0,
        log_potential=0,
        context=(),
        context_ids=(),
        parent=None,
        done=False,
    ):
        self.prompt = prompt
        self.log_weight = log_weight
        self.log_potential = log_potential
        self.context = context
        self.context_ids = context_ids
        self.parent = parent
        self.done = done

    def twist(self, log_potential):
        if self.log_potential == -np.inf:
            assert (
                log_potential == -np.inf
            ), 'Potentials φ must satisfy φ(x) = 0 => φ(xy) = 0, forall x,y in V*'
            self.log_weight = -np.inf
        else:
            self.log_weight += log_potential - self.log_potential
            self.log_potential = log_potential

    def untwist(self):
        self.log_weight -= self.log_potential
        self.log_potential = 0

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        return (
            f'{self.log_weight:.2f}:\t'
            + colors.light.cyan % '['
            + (colors.light.cyan % '|').join(repr(y)[1:-1] for y in self.context)
            + colors.light.cyan % ']'
        )


class ParticleApproximation:
    """
    A particle approximation for a target distribution.

    Attributes:
        particles (list): List of particles representing the distribution.
        size (int): Number of particles in the approximation.
        log_weights (ndarray): Array of log weights for each particle.
        log_total (float): Log of the sum of the weights.
        log_ml (float): Log marginal likelihood estimate.
        log_normalized_weights (ndarray): Array of log-normalized weights.
        log_ess (float): Log of the effective sample size.
        ess (float): Effective sample size, estimated as 1 / sum of squared normalized weights.
        record (SMCRecord or None): Record associated with the approximation, if provided.
    """

    def __init__(self, particles, record=None):
        self.particles = list(particles)
        self.size = len(particles)
        self.log_weights = np.array([p.log_weight for p in self.particles])
        self.log_total = logsumexp(self.log_weights)

        # log-marginal likelihood estimate (Note: need to exponentiate to have
        # an unbiased estimate of the true marginal likelihood).
        self.log_ml = np.log(np.mean(np.exp(self.log_weights)))

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
    return [Particle() for _ in range(n_particles)]


def pretty_print_particles(particles, step_info):
    for i, p in enumerate(particles):
        print(f'├ Particle {i:3d} `{p.context[-1]}` : {p}')
    print(
        f"│ Step {step_info['step']:3d} average weight: {step_info['average_weight']:.4f}"
    )


def stratified_resample(weights):
    # source: https://filterpy.readthedocs.io/en/latest/_modules/filterpy/monte_carlo/resampling.html#stratified_resample
    """Performs the stratified resampling algorithm used by particle filters.

    This algorithms aims to make selections relatively uniformly across the
    particles. It divides the cumulative sum of the weights into N equal
    divisions, and then selects one particle randomly from each division. This
    guarantees that each sample is between 0 and 2/N apart.

    Parameters
    ----------
    weights : list-like of float
        list of weights as floats

    Returns
    -------

    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """

    N = len(weights)
    # make N subdivisions, and chose a random position within each one
    positions = (random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def maybe_resample(
    particles,
    ess_threshold,
    return_record,
    step_info,
    verbosity,
    resample_method='multinomial',
):
    """
    Resamples particles if the effective sample size (ESS) falls below a threshold.

    If a resampling step occurs, all particle weights are set to the average weight of the particles prior to resampling.

    Args:
        particles (list): List of particles.
        ess_threshold (float): Threshold for the effective sample size, specified as a fraction of the number of particles.
        resample_method (str): Resampling method to use. Either 'multinomial' or 'stratified'. Default is 'multinomial'.
        return_record (bool): Flag indicating whether to log additional information in the step_info dictionary.
        step_info (dict): Dictionary to log step information.
        verbosity (int): Verbosity level.

    Returns:
        list: List of resampled particles.

    """
    if return_record:
        step_info['particles'] = [
            {
                'context': p.context,
                'weight': p.log_weight,
                'context_ids': p.context_ids,
            }
            for p in particles
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

    if verbosity > 0:
        pretty_print_particles(particles, step_info)

    if ess < n_particles * ess_threshold:
        if resample_method == 'multinomial':
            indices = log_sample(log_normalized_weights, size=n_particles)
        elif resample_method == 'stratified':
            indices = stratified_resample(np.exp(log_normalized_weights))
        else:
            raise ValueError(f'Unknown resampling method: {resample_method}')

        particles = [
            copy.deepcopy(particles[i])._replace(log_weight=avg_log_weight, parent=i)
            for i in indices
        ]

        if return_record:
            step_info['resample_indices'] = indices.tolist()

        if verbosity > 0:
            print(
                f'└╼  Resampling! {indices}. Weights all set to = {avg_log_weight:.4f}.'
            )
    else:
        if verbosity > 0:
            print('└╼')

    return particles


def twist_particles(particles, potential):
    if potential is None:
        raise ValueError('Potential function must be provided to twist particles.')
    log_potentials = potential(particles)
    for i, p in enumerate(particles):
        p.twist(log_potentials[i])
    return particles


def smc(
    batch_model,
    n_particles,
    potential=None,
    ess_threshold=0.5,
    verbosity=0,
    return_record=False,
    resample_method='multinomial',
):
    """Standard sequential Monte Carlo algorithm with multinomial resampling.

    Args:
      - `batch_model` (`BatchStepper`): The model to perform inference on.
      - `n_particles` (`int`): The number of particles to perform inference with.
      - `potential` (`Callable`):
            A function that when called on a list of particles, returns a list with the log potential values
            for each particle in the list. Optional.
      - `ess_threshold` (`float`): Effective sample size below which resampling
         triggered, given as a fraction of `n_particles`. Default is 0.5.
      - `verbosity` (`int`): Verbosity level. When > 0, particles are printed at each step. Default is 0.
      - `return_record` (`bool`): Flag indicating whether to return a record of the inference steps. Default is False.
      - `resample_method` (`str`): Resampling method to use. Either 'multinomial' or 'stratified'. Default is 'multinomial'.

    Returns:
      - `particle_approximation` (`ParticleApproximation`): The completed particle approximation.
    """

    record = (
        SMCRecord(
            {
                'n_particles': n_particles,
                'ess_threshold': ess_threshold,
                'resample_method': resample_method,
                'algorithm': 'smc_standard_record',
                'history': [],
            }
        )
        if return_record
        else None
    )

    step_num = 1
    particles = init_particles(n_particles)
    while not all(p.done for p in particles):
        step_info = {'step': step_num}

        particles = batch_model.batch_step(particles, is_initial=step_num == 1)

        if (potential is not None) and (ess_threshold > 0):
            particles = twist_particles(particles, potential)

        particles = maybe_resample(
            particles=particles,
            ess_threshold=ess_threshold,
            return_record=return_record,
            step_info=step_info,
            verbosity=verbosity,
            resample_method=resample_method,
        )

        if return_record:
            record['history'].append(step_info)

        step_num += 1

    if (potential is not None) and (ess_threshold == 0):
        particles = twist_particles(particles, potential)

    return ParticleApproximation(particles, record)


def importance_sampling(
    batch_model, n_particles, potential=None, verbosity=0, return_record=False
):
    """Standard sequential importance sampling.

    Args:
      - `batch_model` (`BatchStepper`): The model to perform inference on.
      - `n_particles` (`int`): The number of particles to perform inference with.
      - `potential` (`Callable`):
            A function that when called on a list of particles, returns a list of log potential scores. Optional.
            This potential is only applied at the end of inference.
      - `verbosity` (`int`): Verbosity level. When > 0, particles are printed at each step.

    Returns:
      - `particle_approximation` (`ParticleApproximation`): The completed particle approximation.
    """
    return smc(
        batch_model=batch_model,
        n_particles=n_particles,
        potential=potential,
        ess_threshold=0,
        verbosity=verbosity,
        return_record=return_record,
    )
