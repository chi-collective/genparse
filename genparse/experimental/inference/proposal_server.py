from dataclasses import dataclass
import multiprocessing as mp
import numpy as np
from genparse.util import set_seed
from genparse.lm import LazyProb
from genparse.proposal import CharacterProposal, TokenProposal
import warnings


@dataclass
class Task:
    id: int
    context: str
    logprob_idx: int


@dataclass
class Result:
    id: int
    token: str
    token_id: int
    weight: float


@dataclass
class Error:
    exception: Exception
    id: int


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
        self.eos = self.guide.eos
        self.is_running = False

        self.start()

        print(
            f'Initialized proposal server with num_processes={num_processes}, max_n_particles={max_n_particles}, seed={seed}'
        )

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

        self.is_running = True

    def queue_task(self, id, context, logprob_idx):
        self.task_queue.put(Task(id=id, context=context, logprob_idx=logprob_idx))

    def worker(self, id):
        try:
            set_seed(self.seed + id)

            local_array = np.ndarray(
                (self.max_n_particles, self.llm_vocab_size),
                dtype=np.float32,
                buffer=self.shared_mem.buf,
            )

            proposal = self.create_instance()
        except Exception as e:
            warnings.warn(
                f'Proposal server worker {id} failed during initialization with exception: {e}'
            )
            raise e

        try:
            while True:
                task = self.task_queue.get()
                if task is None:
                    break
                p_llm = LazyProb(
                    _p=np.exp(local_array[task.logprob_idx]),
                    encode=self.llm._encode,
                    decode=self.llm._decode,
                )
                token, _, w = proposal.sample(context=task.context, p_llm=p_llm)
                token_id = (
                    self.llm._encode[token]
                    if not token == self.eos
                    else self.llm.tokenizer.eos_token_id
                )
                self.result_queue.put(
                    Result(id=task.id, token=token, weight=w, token_id=token_id)
                )

        except Exception as e:
            warnings.warn(
                f'Proposal server worker {id} failed while processing a request. Raising error in main process.'
            )
            self.result_queue.put(Error(exception=e, id=id))

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

        self.is_running = False

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
        return TokenProposal(llm=self.llm, guide=self.guide, K=self.K)
