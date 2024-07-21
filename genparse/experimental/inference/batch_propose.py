from collections import namedtuple
from arsenal import colors
import warnings


class Particle(
    namedtuple('Particle', 'prompt, weight, context, context_ids, parent, done')
):
    def __repr__(self):
        return (
            f'{self.weight:.2f}:\t'
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
            weight=particle.weight + w,
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
    def __init__(self, proposal_server, next_token_logprob_server):
        self.proposal_server = proposal_server
        self.next_token_logprob_server = next_token_logprob_server
        self.eos = proposal_server.eos

    def batch_next_token_probs(self, particles, is_initial):
        return self.next_token_logprob_server.execute_request(
            particles=particles, is_initial=is_initial
        )

    def batch_step(self, particles, max_tokens, is_initial=False):
        to_serve = len(particles)

        if to_serve > self.proposal_server.max_n_particles:
            warnings.warn(
                'Num particles to serve > proposal_server.max_n_particles.'
                ' Increasing proposal_server.max_n_particles; allocating more shared memory'
                ' and restarting the proposal server.'
            )
            self.proposal_server.max_n_particles = to_serve
            self.proposal_server.restart()

        logprobs, particle_id_to_logprob_id = self.batch_next_token_probs(
            particles, is_initial
        )

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

            particle = particles[result.id]
            particles[result.id] = Particle(
                prompt=particle.prompt,
                weight=particle.weight + result.weight,
                context=particle.context + (result.token,),
                context_ids=particle.context_ids + (result.token_id,),
                done=(
                    result.token == self.eos or len(particle.context) + 1 >= max_tokens
                ),
                parent=particle.parent,
            )

            to_serve -= 1

        assert self.proposal_server.result_queue.empty(), 'Finished serving particle requests, but result queue is non-empty. Something went wrong :('

        return particles

    def cleanup(self):
        self.proposal_server.cleanup()
        self.next_token_logprob_server.cleanup()
