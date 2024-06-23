import torch

import vllm
from typing import Optional, List, Union
import time
from vllm.engine.output_processor.util import create_output_by_sequence_group
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import Counter
from vllm.usage.usage_lib import UsageContext
from vllm.outputs import (EmbeddingRequestOutput, RequestOutput,
                          RequestOutputFactory)
from vllm.sequence import ExecuteModelRequest
from typing import Sequence as GenericSequence
from typing import Set, Type, TypeVar, Union
from vllm.core.scheduler import (ScheduledSequenceGroup, Scheduler,
                                 SchedulerOutputs)


class LogitsSampler(torch.nn.Module):
    """
        Dummy sampler that returns logits as is.
        Will be called in model_executor.execute_model.
    """

    def __init__(self):
        super().__init__()

        self.include_gpu_probs_tensor = False

    def forward(
        self,
        logits,
        sampling_metadata,
    ):

        logprobs = logits.log_softmax(dim=-1, dtype=torch.float).cpu()

        grouped_logprobs = []
        grouped_seq_ids = []
        sample_idx = 0

        for seq_group in sampling_metadata.seq_groups:

            seq_ids = seq_group.seq_ids
            num_parent_seqs = len(seq_ids)
            grouped_logprobs.append(
                # (seq_bs, vocab_size)
                logprobs[sample_idx:sample_idx+num_parent_seqs]
            )
            grouped_seq_ids.append(
                seq_ids
            )
            sample_idx += num_parent_seqs
        # return logprobs and seq_ids
        # seq_ids denotes which seq_id the first dimension of logprobs corresponds to
        return grouped_logprobs, grouped_seq_ids


class pplLMEngine(vllm.LLMEngine):

    def _process_model_outputs(
        self,
        output,
        scheduled_seq_groups,
        ignored_seq_groups,
        seq_group_metadata_list,
    ):
        """Apply the model output to the sequences in the scheduled seq groups.

        Returns RequestOutputs that can be returned to the client.
        """

        now = time.time()

        # Organize outputs by [sequence group][step] instead of
        # [step][sequence group].

        # Update the scheduled sequence groups with the model outputs.
        for scheduled_seq_group, outputs, seq_group_meta in zip(
                scheduled_seq_groups, output,
                seq_group_metadata_list):
            request_id = seq_group_meta.request_id

            seq_group = scheduled_seq_group.seq_group
            seq_group.update_num_computed_tokens(
                scheduled_seq_group.token_chunk_size)

            self.output_processor.process_prompt_logprob(seq_group, outputs)
            if seq_group_meta.do_sample:
                self.output_processor.process_outputs(seq_group, outputs)

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs = []
        for scheduled_seq_group in scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            seq_group.maybe_set_first_token_time(now)
            request_output = RequestOutputFactory.create(seq_group)
            request_outputs.append(request_output)
        for seq_group in ignored_seq_groups:
            request_output = RequestOutputFactory.create(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    async def next_token_logprobs(self, execute_model_req, **kwargs):
        """
            execute_model_req is the request parameter to the vllm's model_executor
        """

        # logits: list of torch.Tensor each with shape [n_sample, vocab_size]
        # each tensor is the logits for a `sequence_group`
        # n_sample = 1 for in our case
        logits = self.model_executor.execute_model(
            execute_model_req=execute_model_req)

        # cast to float32
        # return: list of torch.Tensor each with shape [n_sample, vocab_size]
        return [l.log_softmax(dim=-1).float() for l in logits]


class vllmpplLLM(vllm.LLM):
    """
        Wrapper around VLLM to make it compatible with hfppl.
            1. vllm sampler replaced with LogitsSampler
            2. added next_token_logprobs, p_next methods
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.llm_engine = pplLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)
        # sampler of the model
        self.llm_engine.model_executor.driver_worker.model_runner.model.sampler = LogitsSampler()
        self.request_counter = Counter()
        self.eos_token_id = self.llm_engine._get_eos_token_id(
            lora_request=None)

    def next_token_logprobs(self, input_ids, **kwargs):
        # call the vllm engine of VLLM
        # kwargs contains the execute_model_req
        if self.input_ids_to_key(input_ids) in self.cache:
            return self.cache[self.input_ids_to_key(input_ids)]
        else:
            batched_log_probs = self.llm_engine.next_token_logprobs(**kwargs)
            self.cache[self.input_ids_to_key(input_ids)] = batched_log_probs

    def p_next(self, input_ids, **kwargs):
        """
            not used in the current implementation
        """
        # kwargs contains the execute_model_req
        return self.next_token_logprobs(input_ids, **kwargs).exp()
