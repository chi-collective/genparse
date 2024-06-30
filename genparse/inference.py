"""
Approximate inference algorithms live here.
"""

import asyncio
import copy

import numpy as np
from arsenal.maths import logsumexp, softmax

from genparse.record import SMCRecord


# _______________________________________________________________________________
#
# The importance sampling method below is an async equivalent of the following
#
# for p in iterview(particles):
#     p.start()
#     while not p.done_stepping():
#         p.step()
#
# There are a few things that we can say about importance sampling
#
# It returns a collection of particles of size `n_particles`.  Each particle in
# this collection has a weight.  That weight must be carefully calculated.
#
#   (i) E[\sum_{y \in P} w(y)] = Z
#
#   (ii) for all y \in universe: E[w(y)] / Z = p(y)
#
#   (iii) for all y \in universe: \lim_{N -> \infty} E[w(y) / \sum_{y \in P} w(y)] = p(y)
#
async def importance_sampling(model, n_particles):
    "Importance sampling estimator"
    # Create n_particles copies of the model
    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    for particle in particles:
        particle.start()
    while not all(p.done_stepping() for p in particles):
        await asyncio.gather(*[p.step() for p in particles if not p.done_stepping()])
    return particles


# _______________________________________________________________________________
# The methods below are borrowed from HFPPL


async def smc_steer(model, n_particles, n_beam):
    """
    Modified sequential Monte Carlo (SMC) algorithm that uses without-replacement resampling,
    as described in [Lew et al. 2023](https://arxiv.org/abs/2306.03081).

    Args:
        model (hfppl.modeling.Model): The model to perform inference on.
        n_particles (int): Number of particles to maintain.
        n_beam (int): Number of continuations to consider for each particle.

    Returns:
        particles (list[hfppl.modeling.Model]): The completed particles after inference.
    """
    verbosity = model.verbosity if hasattr(model, 'verbosity') else 0

    # Create n_particles copies of the model
    particles = [copy.deepcopy(model) for _ in range(n_particles)]

    for particle in particles:
        particle.start()

    if verbosity > 0:
        step_num = 1

    while any(map(lambda p: not p.done_stepping(), particles)):
        # Count the number of finished particles
        n_finished = sum(map(lambda p: p.done_stepping(), particles))
        n_total = n_finished + (n_particles - n_finished) * n_beam

        # Create a super-list of particles that has n_beam copies of each
        super_particles = []
        for p in particles:
            p.untwist()
            super_particles.append(p)
            if p.done_stepping():
                p.weight += np.log(n_total) - np.log(n_particles)
            else:
                p.weight += np.log(n_total) - np.log(n_particles) - np.log(n_beam)
                super_particles.extend([copy.deepcopy(p) for _ in range(n_beam - 1)])

        # Step each super-particle
        await asyncio.gather(
            *[p.step() for p in super_particles if not p.done_stepping()]
        )

        # Use optimal resampling to resample
        weights = np.array([p.weight for p in super_particles])
        total_weight = logsumexp(weights)
        normalized_weights = softmax(weights)
        det_indices, stoch_indices, c = resample_optimal(normalized_weights, n_particles)

        if verbosity > 0:
            for i, p in enumerate(particles):
                print(
                    f'├ Particle {i:3d} (weight {p.weight:.4f}). `{p.context[-1]}` : {p}'
                )
            for i, p in enumerate(super_particles):
                print(
                    f'│├ Super-particle {i:3d} (weight {p.weight:.4f}). `{p.context[-1]}` : {p}'
                )

        particles = [
            super_particles[i] for i in np.concatenate((det_indices, stoch_indices))
        ]

        # For deterministic particles: w = w * N/N'
        for i in det_indices:
            super_particles[i].weight += np.log(n_particles) - np.log(n_total)
        # For stochastic particles: w = 1/c * total       sum(stoch weights) / num_stoch = sum(stoch weights / total) / num_stoch * total * N/M
        for i in stoch_indices:
            super_particles[i].weight = (
                total_weight - np.log(c) + np.log(n_particles) - np.log(n_total)
            )

        if verbosity > 0:
            print(
                '│└ ' f'resample_optimal: det={det_indices}, stoch={stoch_indices}, c={c}'
            )
            for i, p in enumerate(particles):
                print(
                    f'├ Particle {i:3d} (weight {p.weight:.4f}). `{p.context[-1]}` : {p}'
                )
            avg_weight = logsumexp(np.array([p.weight for p in particles])) - np.log(
                n_particles
            )
            print(f'└╼ Step {step_num:3d} average weight: {avg_weight:.4f}')
            step_num += 1

        if verbosity > 0:
            print(
                '│└ ' f'resample_optimal: det={det_indices}, stoch={stoch_indices}, c={c}'
            )
            for i, p in enumerate(particles):
                print(
                    f'├ Particle {i:3d} (weight {p.weight:.4f}). `{p.context[-1]}` : {p}'
                )
            avg_weight = logsumexp(np.array([p.weight for p in particles])) - np.log(
                n_particles
            )
            print(f'└╼ Step {step_num:3d} average weight: {avg_weight:.4f}')
            step_num += 1

    # Return the particles
    return particles


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
    return deterministic, stoch_resampled, c


# _______________________________________________________________________________
#


async def smc_standard(model, n_particles, ess_threshold=0.5):
    """
    Standard sequential Monte Carlo algorithm with multinomial resampling.

    Args:
        model (hfppl.modeling.Model): The model to perform inference on.
        n_particles (int): Number of particles to execute concurrently.
        ess_threshold (float): Effective sample size below which resampling is triggered, given as a fraction of `n_particles`.

    Returns:
        particles (list[hfppl.modeling.Model]): The completed particles after inference.
    """
    verbosity = model.verbosity if hasattr(model, 'verbosity') else 0

    # Create n_particles copies of the model
    particles = [copy.deepcopy(model) for _ in range(n_particles)]

    for particle in particles:
        particle.start()

    if verbosity > 0:
        step_num = 1

    while any(map(lambda p: not p.done_stepping(), particles)):
        # Step each particle
        for p in particles:
            p.untwist()
        await asyncio.gather(*[p.step() for p in particles if not p.done_stepping()])

        # Normalize weights
        weights = np.array([p.weight for p in particles])
        total_weight = logsumexp(weights)
        normalized_weights = weights - total_weight

        if verbosity > 0:
            for i, p in enumerate(particles):
                print(
                    f'├ Particle {i:3d} (weight {p.weight:.4f}). `{p.context[-1]}` : {p}'
                )
            avg_weight = total_weight - np.log(n_particles)
            step_num += 1

        # Resample if necessary
        if -logsumexp(normalized_weights * 2) < np.log(ess_threshold) + np.log(
            n_particles
        ):
            # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
            probs = np.exp(normalized_weights)
            particles = [
                copy.deepcopy(particles[np.random.choice(range(len(particles)), p=probs)])
                for _ in range(n_particles)
            ]
            avg_weight = total_weight - np.log(n_particles)
            for p in particles:
                p.weight = avg_weight

            if verbosity > 0:
                print(f'└╼  Resampling! Weights all set to = {avg_weight:.4f}.')
        else:
            if verbosity > 0:
                print('└╼')

    return particles


# _______________________________________________________________________________
#  Modified version of the above, to keep a record of information about the run.
#  Should be identical


async def smc_standard_record(model, n_particles, ess_threshold=0.5, return_record=True):
    """
    Standard sequential Monte Carlo algorithm with multinomial resampling.

    Args:
        model (hfppl.modeling.Model): The model to perform inference on.
        n_particles (int): Number of particles to execute concurrently.
        ess_threshold (float): Effective sample size below which resampling is triggered, given as a fraction of `n_particles`.

    Returns:
        particles (list[hfppl.modeling.Model]): The completed particles after inference.
        record (SMCRecord): Information about inference run history.
    """
    verbosity = model.verbosity if hasattr(model, 'verbosity') else 0

    # Create n_particles copies of the model
    particles = [copy.deepcopy(model) for _ in range(n_particles)]

    for particle in particles:
        particle.start()

    # Initialize record dict
    record = (
        SMCRecord(
            {
                'step': [0],
                'context': [[p.context.copy() for p in particles]],
                'weight': [[p.weight for p in particles]],
                'resample?': [False],
                'resampled as': [[i for i, _ in enumerate(particles)]],
                'average weight': [0.0],
            }
        )
        if return_record
        else None
    )

    if return_record or verbosity > 0:
        step_num = 1

    while any(map(lambda p: not p.done_stepping(), particles)):
        if return_record:
            record['step'].append(step_num)

        # Step each particle
        for p in particles:
            p.untwist()
        await asyncio.gather(*[p.step() for p in particles if not p.done_stepping()])

        # Normalize weights
        weights = [p.weight for p in particles]
        total_weight = logsumexp(np.array(weights))
        weights_normalized = weights - total_weight

        # Compute log average weight (used if resampling, else only for record)
        avg_weight = total_weight - np.log(n_particles)
        if verbosity > 0:
            for i, p in enumerate(particles):
                print(
                    f'├ Particle {i:3d} (weight {p.weight:.4f}). `{p.context[-1]}` : {p}'
                )
            print(f'│ Step {step_num:3d} average weight: {avg_weight:.4f}')

        if return_record:
            record['context'].append([p.context.copy() for p in particles])
            record['weight'].append(weights)
            record['average weight'].append(avg_weight)

        # Resample if necessary
        if -logsumexp(weights_normalized * 2) < np.log(ess_threshold) + np.log(
            n_particles
        ):
            # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
            probs = np.exp(weights_normalized)

            if return_record or verbosity > 0:
                # resampling: sample indices to copy
                resampled_indices = [
                    np.random.choice(range(len(particles)), p=probs)
                    for _ in range(n_particles)
                ]
                resampled_indices.sort()
                particles = [copy.deepcopy(particles[i]) for i in resampled_indices]
                record['resample?'] += [True]
                record['resampled as'].append(resampled_indices)
            else:
                particles = [
                    copy.deepcopy(
                        particles[np.random.choice(range(len(particles)), p=probs)]
                    )
                    for _ in range(n_particles)
                ]

            for p in particles:
                p.weight = avg_weight

            if verbosity > 0:
                print(
                    f'└╼  Resampling! {resampled_indices}. Weights all set to = {avg_weight:.4f}.'
                )
        else:
            if return_record:
                record['resample?'].append(False)
                record['resampled as'].append([i for i, _ in enumerate(particles)])

            if verbosity > 0:
                print('└╼')

        if return_record or verbosity > 0:
            step_num += 1

    return particles, record
