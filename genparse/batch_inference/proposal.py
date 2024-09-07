import os
import psutil
import warnings
import numpy as np
import multiprocessing as mp
from genparse.lm import LazyProb
from dataclasses import dataclass
from genparse.util import set_seed
from genparse.proposal import CharacterProposal, TokenProposal

MAX_NUM_PROMPTS = 2  # maximum number of distinct prompts


@dataclass
class Task:
    particle_idx: int  # index in `particles` list
    context: str
    logprob_idx: (
        tuple  # 2D index in `ParallelProposal.shared_array` [seq_group_idx, seq_idx]
    )


@dataclass
class Result:
    particle_idx: int
    token: str
    token_id: int
    log_weight: float


@dataclass
class Error:
    exception: Exception
    worker_id: int


class ProposalWorker:
    """
    A worker for parallel inference. Each subprocess contains a ProposalWorker object.

    Args:
        proposal (Proposal): The proposal object.
        shared_array (numpy.ndarray): The shared array of next token log probs.
        worker_id (int): The ID of the worker.
        memory_threshold (float): The memory usage threshold, expressed as a percentage of the total memory
            used by the subprocess, at which to trigger memory management actions.

    Attributes:
        proposal (Proposal): The proposal object.
        shared_array (numpy.ndarray): The shared array of next token log probs.
        worker_id (int): The ID of the worker.
        _decode (list): The decode object from the proposal's llm object. Maps token IDs to tokens.
        _encode (dict): The encode object from the proposal's llm object. Maps tokens to token IDs.
        pid (int): The process ID.
        total_memory (int): The total virtual memory.
        memory_threshold (float): The memory usage threshold, expressed as a percentage of the total memory
            used by the subprocess, at which to trigger memory management actions.

    Methods:
        get_memory_usage: Returns the memory usage of the process.
        maybe_clear_cache: Clears the proposal's cache if memory usage exceeds the threshold.
    """

    def __init__(self, proposal, shared_array, worker_id, memory_threshold):
        self.proposal = proposal
        self.shared_array = shared_array
        self.worker_id = worker_id

        self._decode = self.proposal.llm._decode
        self._encode = self.proposal.llm._encode
        # coerce guide eos to llm eos token id
        self._encode[self.proposal.guide.eos] = self.proposal.llm.tokenizer.eos_token_id

        self.pid = os.getpid()
        self.total_memory = psutil.virtual_memory().total
        self.memory_threshold = memory_threshold

    def get_memory_usage(self):
        """Returns the memory usage of the process, as a percentage of total memory usage."""
        process = psutil.Process(self.pid)
        process_memory = process.memory_info().rss
        memory_percentage = (process_memory / self.total_memory) * 100
        return memory_percentage

    def maybe_clear_cache(self):
        """Clears the guide's cache if memory usage exceeds the threshold."""
        if (self.memory_threshold is not None) and (
            self.get_memory_usage() > self.memory_threshold
        ):
            self.proposal.guide.clear_cache()


def _process_proposal_task(task):
    """
    Function called in subprocesses to process a proposal task and return the result.

    Reads the next token log probabilities from the shared array and samples the next token from the proposal.

    Args:
        task (Task): The proposal task to process.

    Returns:
        Result or Error: The result of processing the task. If an error occurs, an Error object is returned.

    """
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
            particle_idx=task.particle_idx,
            token=token,
            log_weight=w,
            token_id=token_id,
        )
    except Exception as e:
        return Error(exception=e, worker_id=proposal_worker.worker_id)


class ParallelProposal:
    """
    Base class for parallel proposals.

    Attributes:
        llm (LanguageModel): The language model.
        guide (BoolCFGLM): The guide.
        num_processes (int): The number of processes to use during inference.
            A proposal server will be created for each process.
        seed (int): The seed value. Each process will be seeded with a different value.
        memory_threshold (int, optional): The memory usage threshold (as a percentage of total memory) at which
            to trigger memory management actions. Default is 80%.
        eos (str): The guide's end-of-sequence token.
        is_running (bool): Flag indicating if the parallel proposal is running.
        pool (multiprocessing.Pool): The multiprocessing pool.
        max_n_particles (int): The maximum number of particles which can be used during inference. Used to allocate shared memory.
        shared_mem (multiprocessing.shared_memory.SharedMemory): The shared memory for next token logprobs.
            This is used to send logprobs from the main process to the proposal subprocesses.
        shared_array (numpy.ndarray): The shared array of next token logprobs.

    Methods:
        __init__: Initializes the ParallelProposal object.
        _start: Starts the multiprocessing pool and initializes the shared array of next token logprobs.
        _init_worker: Initializes a worker process in the multiprocessing pool.
        batch_particle_extensions: Batch samples particle extensions.
        cleanup: Cleans up the parallel proposal.
        restart: Restarts the parallel proposal.
        create_instance: Creates an instance of the proposal.
            Used by subprocess initilializer to create a proposal in each subprocess.
        __del__: Destructor method that cleans up the parallel proposal.
    """

    def __init__(
        self, llm, guide, num_processes, seed, max_n_particles=250, memory_threshold=70
    ):
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
            size=self.max_n_particles
            * MAX_NUM_PROMPTS
            * llm_vocab_size
            * np.float32().itemsize,
        )

        self.shared_array = np.ndarray(
            (MAX_NUM_PROMPTS, self.max_n_particles, llm_vocab_size),
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

    def batch_particle_extensions(
        self, particles, logprobs_by_seq_group, particle_idx_to_logprob_idx
    ):
        """
        Batch sample particle extensions.

        Args:
            particles (list): A list of particles.
            logprobs (list): A list of next token log probabilities.
            particle_idx_to_logprob_idx (dict): A mapping of particle IDs to next token log probability indices in `logprobs`.

        Returns:
            tuple: A tuple containing the sampled extensions and the mapping of extension indices to their corresponding particle indices.
        """
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

        n_seq_groups, n_seqs, n_tokens = logprobs_by_seq_group.shape
        self.shared_array[:n_seq_groups, :n_seqs, :n_tokens] = logprobs_by_seq_group

        tasks = [
            Task(
                particle_idx=p_idx,
                context=p.context,
                logprob_idx=particle_idx_to_logprob_idx[p_idx],
            )
            for p_idx, p in enumerate(particles)
            if not p.done
        ]

        results = self.pool.map(_process_proposal_task, tasks)

        result_idx_to_particle_idx = np.ones(len(results), dtype=int) * -1
        for i, result in enumerate(results):
            if isinstance(result, Error):
                raise result.exception
            result_idx_to_particle_idx[i] = result.particle_idx

        assert all(result_idx_to_particle_idx != -1)

        return (results, result_idx_to_particle_idx)

    def cleanup(self):
        """
        Cleans up resources used by the parallel proposal.
        If a multiprocessing pool is present, it is closed and joined.
        The shared memory is closed and unlinked.
        """
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

    def batch_particle_extensions(
        self, particles, logprobs_by_seq_group, particle_idx_to_logprob_idx
    ):
        # sequential implementation
        num_extensions = 0
        extensions = []
        extension_id_to_particle_idx = {}
        for particle_idx, p in enumerate(particles):
            if not p.done:
                p_llm = LazyProb(
                    _p=np.exp(
                        logprobs_by_seq_group[particle_idx_to_logprob_idx[particle_idx]]
                    ),
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
                        particle_idx=particle_idx,
                        token=token,
                        log_weight=w,
                        token_id=token_id,
                    )
                )
                extension_id_to_particle_idx[num_extensions] = particle_idx
                num_extensions += 1

        return (extensions, extension_id_to_particle_idx)

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
