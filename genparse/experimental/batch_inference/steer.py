from collections import namedtuple
from arsenal import colors
import numpy as np
from arsenal.timer import Benchmark


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


def init_particles(n_particles):
    return [Particle((), 0, (), (), None, False) for _ in range(n_particles)]


def do_resample(particles, ess_threshold):
    weights = [np.exp(p.log_weight) for p in particles]
    ess = sum(w**2 for w in weights) ** -1
    return ess < ess_threshold


def multinomial_resample(particles):
    weights = [np.exp(p.log_weight) for p in particles]
    probs = weights / sum(weights)
    new_particles_idxs = np.random.choice(particles, size=len(particles), p=probs)
    return [Particle(*particles[i]) for i in new_particles_idxs]


def importance_sampling(prompt, batch_proposal, n_particles, max_tokens):
    particles = init_particles(n_particles)

    batch_proposal.batch_llm.add_prompt(prompt)

    try:
        particles = batch_proposal.batch_step(
            particles, max_tokens=max_tokens, is_initial=True
        )
        while not all(p.done for p in particles):
            particles = batch_proposal.batch_step(particles, max_tokens=max_tokens)
    except Exception as e:
        batch_proposal.cleanup()
        raise e

    return particles


def smc(prompt, batch_proposal, n_particles, max_tokens, ess_threshold=0.5):
    particles = init_particles(n_particles)

    batch_proposal.batch_llm.add_prompt(prompt)

    try:
        particles = batch_proposal.batch_step(
            particles, max_tokens=max_tokens, is_initial=True
        )
        while not all(p.done for p in particles):
            particles = batch_proposal.batch_step(particles, max_tokens=max_tokens)
            if do_resample(particles, ess_threshold):
                particles = multinomial_resample(particles)
    except Exception as e:
        batch_proposal.cleanup()
        raise e

    return particles
