
# TODO:
# 1. replace hfppl_llm with vllm [Done]
# need to implement p_next / next_token_logprobs for vllm
# 2. write VLLMParticle, VLLMSampler in steer.py
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


class LogitsSampler(torch.nn.Module):
    def __init__(self, base_sampler):
        super().__init__()

        self.base_sampler = base_sampler
        self.include_gpu_probs_tensor = False

    def forward(
        self,
        logits,
        sampling_metadata,
    ):
    
        return logits


class pplLMEngine(vllm.LLMEngine):
    async def next_token_logprobs(self, execute_model_req):
        logits = self.model_executor.execute_model(
            execute_model_req=execute_model_req)

        # tensor([[-73.3893, -75.2677, -74.1485,  ..., -81.7179, -80.7128, -69.3829]],
        #        device='cuda:0')
        # SamplerOutput(outputs=[CompletionSequenceGroupOutput(samples=[SequenceOutput(parent_seq_id=0, output_token=33493, logprobs={33493: Logprob(logprob=-0.14365723729133606, rank=1, decoded_token=None)})], prompt_logprobs=None)], sampled_token_probs=None, sampled_token_ids=None, spec_decode_worker_metrics=None)
        
        return logits[0][0].log_softmax(dim=-1).float() #.tolist()
    
    def _process_model_outputs(
            self,
            output,
            scheduled_seq_groups,
            ignored_seq_groups,
            seq_group_metadata_list,
        ):
            now = time.time()
            """Apply the model output to the sequences in the scheduled seq groups.
            Returns RequestOutputs that can be returned to the client.
            """
            # Organize outputs by [sequence group][step] instead of
            # [step][sequence group].
            output_by_sequence_group = create_output_by_sequence_group(
                output, num_seq_groups=len(scheduled_seq_groups))
            # Update the scheduled sequence groups with the model outputs.
            for scheduled_seq_group, outputs, seq_group_meta in zip(
                    scheduled_seq_groups, output_by_sequence_group,
                    seq_group_metadata_list):
                seq_group = scheduled_seq_group.seq_group
                seq_group.update_num_computed_tokens(
                    scheduled_seq_group.token_chunk_size)

                self.output_processor.process_prompt_logprob(seq_group, outputs)
                if seq_group_meta.do_sample:
                    # This is important. Update the generator state with sampled 
                    # token. Should leave the selection of next token to hfppl
                    self.output_processor.process_outputs(seq_group, outputs)

            # Free the finished sequence groups.
            self.scheduler.free_finished_seq_groups()

            # # Create the outputs.
            request_outputs = []
            for scheduled_seq_group in scheduled_seq_groups:
                seq_group = scheduled_seq_group.seq_group
                seq_group.maybe_set_first_token_time(now)
                request_output = RequestOutputFactory.create(seq_group)
                request_outputs.append(request_output)
            for seq_group in ignored_seq_groups:
                request_output = RequestOutputFactory.create(seq_group)
                request_outputs.append(request_output)
            return # request_outputs



class vllmpplLLM(vllm.LLM):

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
        self.llm_engine.model_executor.driver_worker.model_runner.model.sampler = LogitsSampler(
            self.llm_engine.model_executor.driver_worker.model_runner.model.sampler
        )
        self.request_counter = Counter()
        # self.eos = self.llm_engine.model_executor.driver_worker.model_runner.model.eos_token_id
        self.eos_token_id = self.llm_engine._get_eos_token_id(lora_request=None)
        print("self.eos_token_id", self.eos_token_id)


    def next_token_logprobs(self, input_ids, **kwargs): 
        # call the vllm engine of VLLM
        return self.llm_engine.next_token_logprobs(**kwargs)

    def p_next(self, input_ids, **kwargs):
        return self.next_token_logprobs(**kwargs).exp()
