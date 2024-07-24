from dataclasses import dataclass
import multiprocessing as mp
import numpy as np
from genparse.util import set_seed
from genparse.lm import LazyProb
from genparse.proposal import CharacterProposal, TokenProposal
import warnings


@dataclass
class Task:
    particle_id: int
    context: str
    logprob_idx: int


@dataclass
class Result:
    particle_id: int
    token: str
    token_id: int
    log_weight: float


@dataclass
class Error:
    exception: Exception
    worker_id: int


class ParallelProposal:
    def __init__(self, llm, guide, num_processes, max_n_particles, seed):
        self.llm = llm
        self.guide = guide
        self.seed = seed
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.num_processes = num_processes
        self.max_n_particles = max_n_particles
        self.llm_vocab_size = len(llm.V)
        self.eos = self.guide.eos
        self.is_running = False

        self._start()

        print(
            f'Initialized parallel batch proposal with {num_processes=}, {max_n_particles=}, {seed=}'
        )

    def _start(self):
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
            p = mp.Process(target=self._worker, args=(i,))
            p.start()
            self.processes.append(p)

        self.is_running = True

    def _worker(self, worker_id):
        try:
            set_seed(self.seed + worker_id)

            local_array = np.ndarray(
                (self.max_n_particles, self.llm_vocab_size),
                dtype=np.float32,
                buffer=self.shared_mem.buf,
            )

            proposal = self.create_instance()
        except Exception as e:
            warnings.warn(
                f'Proposal server worker {worker_id} failed during initialization with exception: {e}'
            )
            raise e

        try:
            while True:
                # TODO race conditions on get and put from queues?
                task = self.task_queue.get()
                p_llm = LazyProb(
                    _p=np.exp(local_array[task.logprob_idx]),
                    encode=self.llm._encode,
                    decode=self.llm._decode,
                )
                token, _, w = proposal.sample_next_token(
                    context=task.context, p_llm=p_llm
                )
                token_id = (
                    self.llm._encode[token]
                    if not token == self.eos
                    else self.llm.tokenizer.eos_token_id
                )
                self.result_queue.put(
                    Result(
                        particle_id=task.particle_id,
                        token=token,
                        log_weight=w,
                        token_id=token_id,
                    )
                )

        except Exception as e:
            warnings.warn(
                f'Proposal server worker {worker_id} failed while processing a request. Raising error in main process.'
            )
            self.result_queue.put(Error(exception=e, worker_id=worker_id))

    def execute_request(self, particles, logprobs, particle_id_to_logprob_id):
        if not self.is_running:
            warnings.warn(
                'Proposal server is not running (no subprocesses are initialized).'
                ' Attempting to start the server...'
            )
            self.restart()

        to_serve = len([p for p in particles if not p.done])

        if to_serve > self.max_n_particles:
            warnings.warn(
                'Num particles to serve > proposal_server.max_n_particles.'
                ' Increasing proposal_server.max_n_particles; allocating more shared memory'
                ' and restarting the proposal server.'
            )
            self.max_n_particles = to_serve
            self.restart()

        for i, lps in enumerate(logprobs):
            self.shared_array[i] = lps

        for particle_id, p in enumerate(particles):
            if not p.done:
                self.task_queue.put(
                    Task(
                        particle_id=particle_id,
                        context=p.context,
                        logprob_idx=particle_id_to_logprob_id[particle_id],
                    )
                )

        num_results = 0
        results = []
        result_id_to_particle_id = {}
        while to_serve > 0:
            result = self.result_queue.get()

            if isinstance(result, Error):
                self.cleanup()
                raise result.exception

            results.append(result)
            result_id_to_particle_id[num_results] = result.particle_id
            num_results += 1
            to_serve -= 1

        assert self.result_queue.empty(), 'Finished serving particle requests, but result queue is non-empty. Something went wrong :('

        return (results, result_id_to_particle_id)

    def cleanup(self):
        # kill processes
        for p in self.processes:
            p.terminate()
        self.processes = []

        # clean up shared memory
        self.shared_mem.close()
        try:
            self.shared_mem.unlink()
        except FileNotFoundError:
            pass

        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

        self.is_running = False

    def restart(self):
        self.cleanup()
        self._start()

    def create_instance(self):
        raise NotImplementedError('Subclasses must implement the create_instance method')


class ParallelCharacterProposal(ParallelProposal):
    def create_instance(self):
        return CharacterProposal(llm=self.llm, guide=self.guide)


class ParallelTokenProposal(ParallelProposal):
    def __init__(self, K, **kwargs):
        self.K = K
        super().__init__(**kwargs)

    def create_instance(self):
        return TokenProposal(llm=self.llm, guide=self.guide, K=self.K)


#######################
# Sequential baseline #
#######################


class SequentialBatchProposal:
    def __init__(self, llm, guide):
        self.llm = llm
        self.guide = guide
        self.proposal = self.create_instance()
        self.eos = self.proposal.eos

    def execute_request(self, particles, logprobs, particle_id_to_logprob_id):
        # sequential implementation
        num_extensions = 0
        extensions = []
        extension_id_to_particle_id = {}
        for particle_id, p in enumerate(particles):
            if not p.done:
                p_llm = LazyProb(
                    _p=np.exp(logprobs[particle_id_to_logprob_id[particle_id]]),
                    encode=self.llm._encode,
                    decode=self.llm._decode,
                )
                token, _, w = self.proposal.sample_next_token(
                    context=p.context, p_llm=p_llm
                )
                token_id = (
                    self.proposal.llm._encode[token]
                    if not token == self.eos
                    else self.proposal.llm.tokenizer.eos_token_id
                )
                extensions.append(
                    Result(
                        particle_id=particle_id,
                        token=token,
                        log_weight=w,
                        token_id=token_id,
                    )
                )
                extension_id_to_particle_id[num_extensions] = particle_id
                num_extensions += 1

        return (extensions, extension_id_to_particle_id)

    def create_instance(self):
        raise NotImplementedError('Subclasses must implement the create_instance method')


class SequentialCharBatchProposal(SequentialBatchProposal):
    def create_instance(self):
        return CharacterProposal(llm=self.llm, guide=self.guide)


class SequentialTokenBatchProposal(SequentialBatchProposal):
    def __init__(self, K, **kwargs):
        self.K = K
        super().__init__(**kwargs)

    def create_instance(self):
        return TokenProposal(llm=self.llm, guide=self.guide, K=self.K)
