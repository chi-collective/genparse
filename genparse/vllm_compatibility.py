
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
        return logits


class pplLMEngine(vllm.LLMEngine):
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
        return logits[0][0].log_softmax(dim=-1).float()
    

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
        self.eos_token_id = self.llm_engine._get_eos_token_id(lora_request=None)


    def next_token_logprobs(self, input_ids, **kwargs): 
        # call the vllm engine of VLLM
        # kwargs contains the execute_model_req
        return self.llm_engine.next_token_logprobs(**kwargs)


    def p_next(self, input_ids, **kwargs):
        # kwargs contains the execute_model_req
        return self.next_token_logprobs(**kwargs).exp()
