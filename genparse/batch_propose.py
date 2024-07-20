from dataclasses import dataclass
from collections import namedtuple
import multiprocessing as mp
import numpy as np
from arsenal import colors
from genparse.util import set_seed
from genparse.lm import LazyProb
from genparse.proposal import CharacterProposal, TokenProposal
import warnings


class Particle(namedtuple('Particle', 'prompt, weight, context, parent, done')):
    def __repr__(self):
        return (
            f'{self.weight:.2f}:\t'
            + colors.light.cyan % '['
            + (colors.light.cyan % '|').join(repr(y)[1:-1] for y in self.context)
            + colors.light.cyan % ']'
        )


@dataclass
class Task:
    id: int
    context: str


@dataclass
class Result:
    id: int
    token: str
    weight: float


class ProposalServer:
    def __init__(self, llm, guide, num_processes, max_n_particles, seed=0):
        self.llm = llm
        self.guide = guide
        self.seed = seed
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.num_processes = num_processes
        self.max_n_particles = max_n_particles
        self.llm_vocab_size = len(llm.V)

    def start(self):
        # create shared memory
        self.shared_mem = mp.shared_memory.SharedMemory(
            create=True,
            size=self.max_n_particles * self.llm_vocab_size * np.float32().itemsize,
        )

        self.shared_array = np.ndarray(
            (self.max_n_particles, self.llm_vocab_size),
            dtype=np.float32,
            buffer=self.shared_mem.buf,
        )

        self.processes = []
        for i in range(self.num_processes):
            p = mp.Process(target=self.worker, args=(i,))
            p.start()
            self.processes.append(p)

    def worker(self, id):
        set_seed(self.seed + id)

        local_array = np.ndarray(
            (self.max_n_particles, self.llm_vocab_size),
            dtype=np.float32,
            buffer=self.shared_mem.buf,
        )

        proposal = self.create_instance()

        while True:
            task = self.task_queue.get()
            if task is None:
                break

            p_llm = LazyProb(
                _p=local_array[task.id], encode=self.llm._encode, decode=self.llm._decode
            )
            token, _, w = proposal.sample(context=task.context, p_llm=p_llm)
            self.result_queue.put(Result(id=task.id, token=token, weight=w))

    def cleanup(self):
        # kill processes
        for p in self.processes:
            p.terminate()
        self.processes = []

        # clean up shared memory
        self.shared_mem.close()
        self.shared_mem.unlink()

    def restart(self):
        self.cleanup()
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.start()


class CharacterProposalServer(ProposalServer):
    def create_instance(self):
        return CharacterProposal(self.llm, self.guide)


class TokenProposalServer(ProposalServer):
    def __init__(self, K, **kwargs):
        self.K = K
        super().__init__(**kwargs)

    def create_instance(self):
        return TokenProposal(self.llm, self.guide, self.K)


class BatchProposal:
    def __init__(self, proposal_server, eos=None):
        self.proposal_server = proposal_server
        self.eos = eos if eos is not None else proposal_server.guide.eos

    def batch_next_token_probs(self, particles):
        # default sequential implementation
        return [
            self.proposal_server.llm.p_next(p.prompt + p.context) if not p.done else None
            for p in particles
        ]

    def batch_step(self, particles, max_tokens):
        to_serve = len(particles)

        if to_serve > self.proposal_server.max_n_particles:
            warnings.warn(
                'Num particles to serve > proposal_server.max_n_particles.'
                ' Increasing proposal_server.max_n_particles; allocating more shared memory'
                ' and restarting the proposal server.'
            )
            self.proposal_server.max_n_particles = to_serve
            self.proposal_server.restart()

        p_llms = self.batch_next_token_probs(particles)

        for i, p in enumerate(particles):
            if not p.done:
                # write logprobs to shared array
                self.proposal_server.shared_array[i] = p_llms[i]._p
                # queue task for subprocesses
                self.proposal_server.task_queue.put(Task(id=i, context=p.context))
            else:
                to_serve -= 1

        while to_serve > 0:
            result = self.proposal_server.result_queue.get()

            particle = particles[result.id]
            particles[result.id] = Particle(
                prompt=particle.prompt,
                weight=particle.weight + result.weight,
                context=particle.context + (result.token,),
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
        # self.next_token_logprob_server.cleanup()


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

        return Particle(
            prompt=particle.prompt,
            weight=particle.weight + w,
            context=particle.context + (token,),
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
