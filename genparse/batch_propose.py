from dataclasses import dataclass
import multiprocessing as mp
import numpy as np
from genparse.util import set_seed
from genparse.lm import LazyProb
from genparse.proposal import CharacterProposal, TokenProposal


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

        # self.start()

    def start(self):
        # Create shared memory
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

    def cleanup(self):
        # kill processes
        for p in self.processes:
            p.terminate()

        # clean up shared memory
        self.shared_mem.close()
        self.shared_mem.unlink()

    def restart(self):
        self.cleanup()
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.start()

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
    def __init__(self, proposal_server):
        self.proposal_server = proposal_server

    def batch_next_token_probs(self, particles):
        # default sequential implementation
        return [self.proposal_server.llm.p_next(p.prompt + p.context) for p in particles]

    def batch_step(self, particles):
        to_serve = len(particles)

        if to_serve > self.proposal_server.max_n_particles:
            raise ValueError(
                'Too many particles. Consider increasing self.proposal_server.max_n_particles.'
            )

        p_llms = self.batch_next_token_probs(particles)

        for i, p_llm in enumerate(p_llms):
            self.proposal_server.shared_array[i] = p_llm._p

        for i, p in enumerate(particles):
            if not p.is_done():
                self.proposal_server.task_queue.put(Task(id=i, context=p.context))
            else:
                to_serve -= 1

        while to_serve > 0:
            result = self.proposal_server.result_queue.get()
            particle = particles[result.id]
            particle.context += (result.token,)
            particle.weight += result.weight
            particles[result.id] = particle
            to_serve -= 1

        if not self.proposal_server.result_queue.empty():
            raise ValueError(
                'Finished serving particle requests, but result queue is non-empty.'
            )

        return particles

    def cleanup(self):
        self.proposal_server.cleanup()


class BatchProposalBaseline:
    def __init__(self, proposal):
        self.proposal = proposal

    def batch_next_token_probs(self, particles):
        # default sequential implementation
        return [self.proposal.llm.p_next(p.prompt + p.context) for p in particles]

    def individual_step(self, particle, p_llm):
        token, _, w = self.proposal.sample(context=particle.context, p_llm=p_llm)
        particle.context += (token,)
        particle.weight += w
        return particle

    def batch_step(self, particles):
        p_llms = self.batch_next_token_probs(particles)
        # default sequential implementation
        particles = [
            self.individual_step(p, p_llm) if not p.is_done() else p
            for p, p_llm in zip(particles, p_llms)
        ]
        return particles
