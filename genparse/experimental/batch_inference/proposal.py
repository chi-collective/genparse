import os
import psutil
import warnings
import numpy as np
import multiprocessing as mp
from genparse.lm import LazyProb
from dataclasses import dataclass
from genparse.util import set_seed
from genparse.proposal import CharacterProposal, TokenProposal


@dataclass
class Task:
    particle_id: int
    context: str
    logprob_idx: int  # index in `ParallelProposal.shared_array`


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


class ProposalWorker:
    def __init__(self, proposal, shared_array, worker_id, memory_threshold):
        self.proposal = proposal
        self.shared_array = shared_array
        self.worker_id = worker_id

        self._decode = self.proposal.llm._decode
        self._encode = self.proposal.llm._encode
        self._encode[self.proposal.guide.eos] = self.proposal.llm.tokenizer.eos_token_id

        self.pid = os.getpid()
        self.total_memory = psutil.virtual_memory().total
        self.memory_threshold = memory_threshold

    def get_memory_usage(self):
        process = psutil.Process(self.pid)
        process_memory = process.memory_info().rss
        memory_percentage = (process_memory / self.total_memory) * 100
        return memory_percentage

    def maybe_clear_cache(self):
        if (self.memory_threshold is not None) and (
            self.get_memory_usage() > self.memory_threshold
        ):
            self.proposal.guide.clear_cache()


def _process_proposal_task(task):
    try:
        p_llm = LazyProb(
            _p=np.exp(proposal_worker.shared_array[task.logprob_idx]),
            encode=proposal_worker._encode,
            decode=proposal_worker._decode,
        )
        token, _, w = proposal_worker.proposal.sample_next_token_sync(
            context=task.context, p_llm=p_llm
        )
        token_id = proposal_worker._encode[token]

        proposal_worker.maybe_clear_cache()

        return Result(
            particle_id=task.particle_id,
            token=token,
            log_weight=w,
            token_id=token_id,
        )
    except Exception as e:
        return Error(exception=e, worker_id=proposal_worker.worker_id)


class ParallelProposal:
    def __init__(
        self, llm, guide, num_processes, seed, max_n_particles=250, memory_threshold=80
    ):
        """
        Args:
            llm: .
            guide (BoolCFGLM): .
            num_processes (int): The number of parallel processes to use during batch inference.
            max_n_particles (int): The maximum number of particles.
            seed (int): The seed value.
            memory_threshold (int, optional): The memory usage threshold (as a percentage of total memory) at which
                to trigger memory management actions. Default is 80%.

        Initializes the multiprocessing pool and shared array of logprobs.
        """
        self.llm = llm
        self.guide = guide
        self.seed = seed
        self.num_processes = num_processes
        self.max_n_particles = max_n_particles
        self.eos = self.guide.eos
        self.is_running = False
        self.pool = None
        self.memory_threshold = memory_threshold

        self._start()

    def _start(self):
        llm_vocab_size = len(self.llm.V)

        self.shared_mem = mp.shared_memory.SharedMemory(
            create=True,
            size=self.max_n_particles * llm_vocab_size * np.float32().itemsize,
        )

        self.shared_array = np.ndarray(
            (self.max_n_particles, llm_vocab_size),
            dtype=np.float32,
            buffer=self.shared_mem.buf,
        )

        manager = mp.Manager()
        id_queue = manager.Queue()
        for i in range(self.num_processes):
            id_queue.put(i)

        self.pool = mp.Pool(
            processes=self.num_processes,
            initializer=self._init_worker,
            initargs=(id_queue, self.memory_threshold / self.num_processes),
        )

        self.is_running = True

    def _init_worker(self, id_queue, memory_threshold):
        worker_id = id_queue.get()
        set_seed(abs(hash((self.seed, worker_id)) % 2**32))

        global proposal_worker
        proposal_worker = ProposalWorker(
            self.create_instance(), self.shared_array, worker_id, memory_threshold
        )

    def batch_particle_extensions(self, particles, logprobs, particle_id_to_logprob_id):
        if not self.is_running:
            warnings.warn(
                'Proposal server is not running (pool is not initialized).'
                ' Attempting to initialize the pool...'
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

        tasks = [
            Task(
                particle_id=p_idx,
                context=p.context,
                logprob_idx=particle_id_to_logprob_id[p_idx],
            )
            for p_idx, p in enumerate(particles)
            if not p.done
        ]

        results = self.pool.map(_process_proposal_task, tasks)

        result_id_to_particle_idx = {}
        for i, result in enumerate(results):
            if isinstance(result, Error):
                raise result.exception
            result_id_to_particle_idx[i] = result.particle_id

        return (results, result_id_to_particle_idx)

    def cleanup(self):
        if self.pool is not None:
            try:
                self.pool.close()
                self.pool.join()
            except KeyboardInterrupt:
                self.pool.terminate()
                self.pool.join()
            self.pool = None

        self.shared_mem.close()
        try:
            self.shared_mem.unlink()
        except FileNotFoundError:
            pass

        self.is_running = False

    def restart(self):
        self.cleanup()
        self._start()

    def create_instance(self):
        raise NotImplementedError('Subclasses must implement the create_instance method')

    def __del__(self):
        self.cleanup()


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

    def batch_particle_extensions(self, particles, logprobs, particle_id_to_logprob_id):
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
                token, _, w = self.proposal.sample_next_token_sync(
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

    def cleanup(self):
        self.guide.clear_cache()


class CharacterBatchProposal(SequentialBatchProposal):
    def create_instance(self):
        return CharacterProposal(llm=self.llm, guide=self.guide)


class TokenBatchProposal(SequentialBatchProposal):
    def __init__(self, K, **kwargs):
        self.K = K
        super().__init__(**kwargs)

    def create_instance(self):
        return TokenProposal(llm=self.llm, guide=self.guide, K=self.K)
