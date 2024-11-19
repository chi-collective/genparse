import copy
import atexit
import warnings
import numpy as np
from arsenal import colors
from genparse import Float
from numpy.random import random
from collections import namedtuple
from genparse.record import SMCRecord
from arsenal.maths import logsumexp, log_sample, sample_dict


class BatchStepModel:
    """
    A model for performing batch steps during inference.

    Attributes:
        batch_proposal (BatchProposal): The batch proposal.
        batch_llm (BatchLLM): The batch language model.
        max_tokens (int): The maximum number of tokens to sample.

    Methods:
        set_prompt: Set the prompt for the batch language model.
        batch_step: Perform a batch step during inference.
        cleanup: Clean up resources used by the batch proposal and batch language model.

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

    def batch_step(self, particles, is_initial=False, free_dead_seqs=True):
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
                particles=particles, is_initial=is_initial, free_dead_seqs=free_dead_seqs
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


def twist_particles(particles, potential, step_num):
    if potential is None:
        raise ValueError('Potential function must be provided to twist particles.')
    log_potentials = potential(particles=particles, step_num=step_num)
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
            for each particle in the list of particles. Optional.
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
            particles = twist_particles(particles, potential, step_num)

        particles = maybe_resample(
            particles=particles,
            ess_threshold=ess_threshold,
            return_record=return_record,
            step_info=step_info,
            verbosity=verbosity,
            resample_method=resample_method,
        )

        if verbosity > 0:
            pretty_print_particles(particles, step_info)

        if return_record:
            record['history'].append(step_info)

        step_num += 1

    if (potential is not None) and (ess_threshold == 0):
        particles = twist_particles(particles, potential, step_num)

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


########################
# SMC steering methods #
########################


def find_c(weights, N):
    # Sort the weights
    sorted_weights = np.sort(weights)
    # Find the smallest chi
    B_val = 0.0
    A_val = len(weights)
    for i in range(len(sorted_weights)):
        chi = sorted_weights[i]
        # Calculate A_val -- number of weights larger than chi
        A_val -= 1
        # Update B_val -- add the sum of weights smaller than or equal to chi
        B_val += chi
        if B_val / chi + A_val - N <= 1e-12:
            return (N - A_val) / B_val
    return N


def resample_optimal(weights, N):
    c = find_c(weights, N)
    # Weights for which c * w >= 1 are deterministically resampled
    deterministic = np.where(c * weights >= 1)[0]
    # Weights for which c * w <= 1 are stochastically resampled
    stochastic = np.where(c * weights < 1)[0]
    # Stratified sampling to generate N-len(deterministic) indices
    # from the stochastic weights
    n_stochastic = len(stochastic)
    n_resample = N - len(deterministic)
    if n_resample == 0:
        return deterministic, np.array([], dtype=int), c
    K = np.sum(weights[stochastic]) / (n_resample)
    u = np.random.uniform(0, K)
    i = 0
    stoch_resampled = np.array([], dtype=int)
    while i < n_stochastic:
        u = u - weights[stochastic[i]]
        if u <= 0:
            # Add stochastic[i] to resampled indices
            stoch_resampled = np.append(stoch_resampled, stochastic[i])
            # Update u
            u = u + K
            i = i + 1
        else:
            i += 1
    # Concatenate the deterministic and stochastic resampled indices
    # resampled = np.concatenate((deterministic, stoch_resampled))
    # return resampled
    return deterministic, stoch_resampled, c


def log_softmax(nums):
    """Compute log(softmax(nums)).

    Args:
        nums: a vector or numpy array of unnormalized log probabilities.

    Returns:
        np.array: an array of log (normalized) probabilities.
    """
    return nums - logsumexp(nums)


def softmax(nums):
    return np.exp(log_softmax(nums))


def smc_steer(
    batch_model, n_particles, n_beam, potential=None, return_record=False, verbosity=0
):
    record = (
        SMCRecord(
            {
                'n_particles': n_particles,
                'n_beam': n_beam,
                'algorithm': 'smc_steer_record',
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

        # Count the number of finished particles
        n_finished = sum(map(lambda p: p.done, particles))
        n_total = n_finished + (n_particles - n_finished) * n_beam

        # Create a super-list of particles that has n_beam copies of each
        super_particles = []
        for p in particles:
            super_particles.append(p)
            if p.done:
                p.log_weight += np.log(n_total) - np.log(n_particles)
            else:
                p.log_weight += np.log(n_total) - np.log(n_particles) - np.log(n_beam)
                super_particles.extend([copy.deepcopy(p) for _ in range(n_beam - 1)])

        super_particles = batch_model.batch_step(
            super_particles, is_initial=step_num == 1
        )

        if potential is not None:
            super_particles = twist_particles(super_particles, potential, step_num)

        # Use optimal resampling to resample
        W = np.array([p.log_weight for p in super_particles])
        W_tot = logsumexp(W)
        W_normalized = softmax(W)
        det_indices, stoch_indices, c = resample_optimal(W_normalized, n_particles)
        particles = [
            super_particles[i] for i in np.concatenate((det_indices, stoch_indices))
        ]
        # For deterministic particles: w = w * N/N'
        for i in det_indices:
            super_particles[i].log_weight += np.log(n_particles) - np.log(n_total)
        # For stochastic particles: w = 1/c * total       sum(stoch weights) / num_stoch = sum(stoch weights / total) / num_stoch * total * N/M
        for i in stoch_indices:
            super_particles[i].log_weight = (
                W_tot - np.log(c) + np.log(n_particles) - np.log(n_total)
            )

        if verbosity > 0:
            pretty_print_particles(particles, step_info)

        if return_record:
            step_info['particles'] = [
                {
                    'context': p.context,
                    'weight': p.log_weight,
                    'context_ids': p.context_ids,
                }
                for p in particles
            ]
            record['history'].append(step_info)

        step_num += 1

    if potential is not None:
        particles = twist_particles(particles, potential, step_num)

    return ParticleApproximation(particles, record)
