from collections import namedtuple
from arsenal import colors
import warnings
import numpy as np
from arsenal.timer import Benchmark
from genparse.experimental.inference.proposal_server import Error


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


class BatchProposalBaseline:
    def __init__(self, proposal, clear_cache=False):
        self.proposal = proposal
        self.eos = self.proposal.eos
        self.clear_cache = clear_cache

    def batch_next_token_probs(self, particles):
        # default sequential implementation
        return [
            self.proposal.llm.p_next(p.prompt + p.context) if not p.done else None
            for p in particles
        ]

    def individual_step(self, particle, p_llm, max_tokens):
        if self.clear_cache:
            self.proposal.guide.model._chart.clear()

        token, _, w = self.proposal.sample(context=particle.context, p_llm=p_llm)
        token_id = self.proposal.llm._encode[token]

        return Particle(
            prompt=particle.prompt,
            weight=particle.log_weight + w,
            context=particle.context + (token,),
            context_ids=particle.context_ids + (token_id,),
            done=(token == self.eos or len(particle.context) + 1 >= max_tokens),
            parent=particle.parent,
        )

    def batch_step(self, particles, max_tokens):
        p_llms = self.batch_next_token_probs(particles)
        # default sequential implementation
        particles = [
            self.individual_step(p, p_llm, max_tokens) if not p.done else p
            for p, p_llm in zip(particles, p_llms)
        ]
        return particles


class BatchProposal:
    def __init__(self, proposal_server, next_token_logprob_server, max_tokens):
        self.proposal_server = proposal_server
        self.next_token_logprob_server = next_token_logprob_server
        self.eos = proposal_server.eos
        self.max_tokens = max_tokens
        self.timer = Benchmark('VLLM vs CFG+trie+weight')

        print(
            f'Initialized batch proposal with eos={self.eos} and max_tokens={self.max_tokens}'
        )

    def batch_next_token_probs(self, particles, is_initial):
        return self.next_token_logprob_server.execute_request(
            particles=particles, is_initial=is_initial
        )

    def batch_extend_particles(self, particles, logprobs, particle_id_to_logprob_id):
        to_serve = len(particles)

        if to_serve > self.proposal_server.max_n_particles:
            warnings.warn(
                'Num particles to serve > proposal_server.max_n_particles.'
                ' Increasing proposal_server.max_n_particles; allocating more shared memory'
                ' and restarting the proposal server.'
            )
            self.proposal_server.max_n_particles = to_serve
            self.proposal_server.restart()

        for i, lps in enumerate(logprobs):
            self.proposal_server.shared_array[i] = lps

        for i, p in enumerate(particles):
            if not p.done:
                self.proposal_server.queue_task(
                    id=i, context=p.context, logprob_idx=particle_id_to_logprob_id[i]
                )
            else:
                to_serve -= 1

        while to_serve > 0:
            result = self.proposal_server.result_queue.get()

            if isinstance(result, Error):
                self.cleanup()
                raise result.exception

            particle = particles[result.id]
            particles[result.id] = Particle(
                prompt=particle.prompt,
                log_weight=particle.log_weight + result.weight,
                context=particle.context + (result.token,),
                context_ids=particle.context_ids + (result.token_id,),
                done=(
                    result.token == self.eos
                    or len(particle.context) + 1 >= self.max_tokens
                ),
                parent=particle.parent,
            )

            to_serve -= 1

        assert self.proposal_server.result_queue.empty(), 'Finished serving particle requests, but result queue is non-empty. Something went wrong :('

        return particles

    def batch_step(self, particles, is_initial=False):
        if not self.proposal_server.is_running:
            raise ValueError(
                'Proposal server is not running (no subprocesses are initialized). Please start the server.'
            )

        with self.timer['vllm']:
            logprobs, particle_id_to_logprob_id = self.batch_next_token_probs(
                particles, is_initial
            )

        with self.timer['parser']:
            particles = self.batch_extend_particles(
                particles, logprobs, particle_id_to_logprob_id
            )

        return particles

    def cleanup(self):
        self.proposal_server.cleanup()
        self.next_token_logprob_server.cleanup()


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

    batch_proposal.next_token_logprob_server.add_prompt(prompt)

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

    batch_proposal.next_token_logprob_server.add_prompt(prompt)

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
