import sys
import torch
import warnings
import numpy as np
from collections import defaultdict
from genparse.lm import VirtualTokenizedLLM
from genparse.util import load_model_by_name

try:
    import vllm

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class BatchLLM:
    """Simple class which wraps LMs. Next token logprobs are computed sequentially from the llm."""

    def __init__(self, llm):
        self.llm = llm
        self.eos = self.llm.eos
        self.eos_token_id = self.llm._encode[self.eos]

    @classmethod
    def from_name(cls, model_name):
        return cls(llm=load_model_by_name(model_name))

    def set_prompt(self, prompt):
        self.prompt = self.llm.encode_prompt(prompt)

    def batch_next_token_logprobs(self, particles, *args, **kwargs):
        logprobs = []
        particle_id_to_logprob_idx = {}
        for i, p in enumerate(particles):
            if not p.done:
                logprob = self.llm.logp_next(self.prompt + p.context)
                logprobs.append(logprob._p)
                particle_id_to_logprob_idx[i] = len(logprobs) - 1

        return logprobs, particle_id_to_logprob_idx

    def cleanup(self):
        pass


class LogitsGrouper(torch.nn.Module):
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
            grouped_logprobs.append(logprobs[sample_idx : sample_idx + num_parent_seqs])
            grouped_seq_ids.append(seq_ids)
            sample_idx += num_parent_seqs

        return grouped_logprobs, grouped_seq_ids


class VLLMParticleMetadata:
    def __init__(self):
        self.sequence_id_to_particle_ids = {}
        self.scheduler_outputs = None
        self.seq_group_metadata_list = None
        self.sequence_ids_by_seq_group = None


if VLLM_AVAILABLE:
    from vllm.sequence import (
        Logprob,
        SamplerOutput,
        SequenceOutput,
        CompletionSequenceGroupOutput,
    )
    from vllm.utils import Counter
    from vllm import SamplingParams
    from vllm.sequence import ExecuteModelRequest
    from vllm.engine.output_processor.util import create_output_by_sequence_group

    class BatchVLLM(vllm.LLM):
        """
        Batch LM sampling with VLLM.

        This class is a wrapper around VirtualTokenizedLLMs. It provides an interface for
        batched next token logprob computations with the VLLM engine.

        Attributes:
            llm (VirtualTokenizedLLM): The VirtualTokenizedLLM instance.
            llm_engine (VLLMEngine): The VLLM engine.
            eos (str): The end-of-sequence token.
            eos_token_id (int): The end-of-sequence token id.
            request_counter (Counter): A counter for tracking the number of requests.
            particle_metadata (VLLMParticleMetadata):
                Metadata for tracking VLLM sequence information associated with the particle state.
            prompt (str): The LM prompt.

        Methods:
            from_name: Create a BatchVLLM instance from a HF model name.
            set_prompt: Set the LM prompt.
            _make_initial_request: Make an initial request to the VLLM engine.
            batch_next_token_logprobs: Compute the next token logprobs for a batch of particles.
            _map_sequence_id_to_particle_ids: Associate the scheduled sequence ids in the VLLM with particle ids.
            _register_particle_extensions: Update the VLLM engine with the sampled particle extensions.
            free_unfinished_requests: Free all requests from the VLLM engine.
            cleanup: Cleanup the BatchVLLM instance.
        """

        def __init__(self, llm):
            if not isinstance(llm, VirtualTokenizedLLM):
                raise ValueError('BatchVLLM requires a VirtualTokenizedLLM instance.')
            self.llm = llm
            self.llm_engine = self.llm.llm_engine
            self.llm_engine.model_executor.driver_worker.model_runner.model.sampler = (
                LogitsGrouper()
            )
            self.eos = self.llm.eos
            self.eos_token_id = self.llm._encode[self.eos]
            self.request_counter = Counter()
            self.particle_metadata = VLLMParticleMetadata()
            self.prompt = None

            assert (
                self.eos_token_id == self.llm_engine.tokenizer.tokenizer.eos_token_id
            ), (
                'BatchVLLM eos_token misalignment; '
                f'eos_token_id ({self.eos_token_id}) != vllm engine eos_token_id ({self.llm_engine.tokenizer.tokenizer.eos_token_id})'
                'This will cause issues with particle termination conditions.'
            )

        def free_vllm_gpu_memory(self):
            """
            Frees any residual GPU memory that VLLM allocated.

            Calling this method is only necessary if you will go on to use the GPU later in your code. The
            GPU memory will be deallocated as normal when the Python process ends. This method allows freeing the GPU
            memory early.

            Note that this batch LLM object will become unusable after calling this method. Call this when and only when
            you're finished with the batc hLLM object.
            """
            self.llm.free_vllm_gpu_memory()

        @classmethod
        def from_name(cls, model_name):
            return cls(llm=VirtualTokenizedLLM.from_name(model_name))

        def set_prompt(self, prompt):
            self.prompt = prompt

        def _make_initial_request(self, prompt):
            self._validate_and_add_requests(
                inputs=self._convert_v1_inputs(prompts=prompt, prompt_token_ids=None),
                params=SamplingParams(
                    max_tokens=np.inf, stop_token_ids=[self.eos_token_id], stop=None
                ),
                lora_request=None,
            )

        def batch_next_token_logprobs(self, particles, is_initial=False):
            """Take a single VLLM step to compute logprobs for the next token"""
            if is_initial:
                if self.llm_engine.has_unfinished_requests():
                    # 'Engine has unfinished requests from previous runs. Freeing leftover requests.'
                    self.free_unfinished_requests()

                if self.prompt is None:
                    raise ValueError('Initial request requires a prompt.')

                self._make_initial_request(prompt=self.prompt)
            else:
                self._map_sequence_id_to_particle_ids(
                    particles, from_possible_resampling=True
                )
                # register particle state with the VLLM Engine
                self._register_particle_extensions(particles)

            seq_group_metadata_list, scheduler_outputs = (
                self.llm_engine.scheduler.schedule()
            )

            assert len(seq_group_metadata_list) == 1, (
                'There should only be a single sequence group'
            )

            # update particle metadata with scheduler metadata and output
            self.particle_metadata.scheduler_outputs = scheduler_outputs
            self.particle_metadata.seq_group_metadata_list = seq_group_metadata_list
            self._map_sequence_id_to_particle_ids(
                particles, from_possible_resampling=False
            )

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
            )

            logprobs_by_step, sequence_ids_by_step = zip(
                *self.llm_engine.model_executor.execute_model(
                    execute_model_req=execute_model_req
                )
            )

            # [sequence group][step]
            logprobs_by_seq_group = create_output_by_sequence_group(
                logprobs_by_step, num_seq_groups=1
            )
            sequence_ids_by_seq_group = create_output_by_sequence_group(
                sequence_ids_by_step, num_seq_groups=1
            )

            self.particle_metadata.sequence_ids_by_seq_group = sequence_ids_by_seq_group

            assert len(logprobs_by_seq_group) == 1, (
                'There should only be one sequence group (logprobs)'
            )
            assert len(logprobs_by_seq_group[0]) == 1, (
                'We should only be decoding a single step (logprobs)'
            )
            assert len(sequence_ids_by_seq_group) == 1, (
                'There should only be one sequence group (sequence ids)'
            )
            assert len(sequence_ids_by_seq_group[0]) == 1, (
                'We should only be decoding a single step (sequence ids)'
            )

            logprobs = logprobs_by_seq_group[0][0]
            sequence_ids = sequence_ids_by_seq_group[0][0]

            particle_id_to_logprob_idx = {
                particle_id: sequence_ids.index(seq_id)
                for seq_id, particle_ids in self.particle_metadata.sequence_id_to_particle_ids.items()
                for particle_id in particle_ids
            }

            return logprobs.numpy(), particle_id_to_logprob_idx

        def _map_sequence_id_to_particle_ids(
            self, particles, from_possible_resampling=False
        ):
            """
            Associate the scheduled sequence ids in the VLLM engine with particle ids

            Args:
                particles (list): A list of particles.
                from_possible_resampling (bool): Whether a resampling step might have occured.
                    When true, particles and VLLM sequences are realigned excluding the most recent token in particle context.
                    This step must be done prior to registering particle extensions.
            """

            # TODO: this can be optimized; the from_possible_resampling = True case is a hack to remap sequence ids to particle ids
            # in case there was a resampling step

            seq_group_metadata = self.particle_metadata.seq_group_metadata_list[0]

            context_ids_to_particle_ids = {
                tuple(seq_data.output_token_ids): []
                for seq_data in seq_group_metadata.seq_data.values()
            }
            for particle_id, particle in enumerate(particles):
                if not particle.done:
                    context_ids = (
                        particle.context_ids[:-1]
                        if from_possible_resampling
                        else particle.context_ids
                    )
                    try:
                        context_ids_to_particle_ids[context_ids].append(particle_id)
                    except KeyError:
                        # This KeyError may arise if context_ids[-1] == self.llm_engine.tokenizer.tokenizer.eos_token_id,
                        # but particle.done = False. In those cases, the VLLM scheduler will not schedule sequences which
                        # end in the EOS token.
                        raise KeyError(
                            'Particle context ids not found in seq group metadata.'
                        )

            sequence_id_to_particle_ids = {}
            for seq_id, seq_data in seq_group_metadata.seq_data.items():
                particles_ids = context_ids_to_particle_ids[
                    tuple(seq_data.output_token_ids)
                ]
                sequence_id_to_particle_ids[seq_id] = particles_ids

            self.particle_metadata.sequence_id_to_particle_ids = (
                sequence_id_to_particle_ids
            )

        def _register_particle_extensions(self, particles):
            """
            Update the VLLM Engine with the sampled particle extensions.

            This function updates the sequence outputs for each sequence id with the sample particle extensions.

            Args:
                particles (list): A list of particles.
            """

            sequence_outputs = []
            for (
                parent_seq_id,
                particle_ids,
            ) in self.particle_metadata.sequence_id_to_particle_ids.items():
                unique_token_ids = []
                for particle_id in particle_ids:
                    sampled_token_id = particles[particle_id].context_ids[-1]

                    if sampled_token_id not in unique_token_ids:
                        unique_token_ids.append(sampled_token_id)
                        sequence_outputs.append(
                            SequenceOutput(
                                parent_seq_id=parent_seq_id,
                                output_token=sampled_token_id,
                                logprobs={sampled_token_id: Logprob(logprob=0)},
                            )
                        )

            sampler_outputs = [
                SamplerOutput(
                    outputs=[
                        CompletionSequenceGroupOutput(
                            samples=sequence_outputs, prompt_logprobs=None
                        )
                    ]
                )
            ]

            scheduler_outputs = self.particle_metadata.scheduler_outputs
            seq_group_metadata_list = self.particle_metadata.seq_group_metadata_list

            self.llm_engine._process_model_outputs(
                sampler_outputs,
                scheduler_outputs.scheduled_seq_groups,
                scheduler_outputs.ignored_seq_groups,
                seq_group_metadata_list,
            )

        def free_unfinished_requests(self):
            """Free all requests from the VLLM Engine."""
            for group in list(self.llm_engine.scheduler.running):
                self.llm_engine.abort_request(group.request_id)

            for group in list(self.llm_engine.scheduler.waiting):
                self.llm_engine.abort_request(group.request_id)

            for group in list(self.llm_engine.scheduler.swapped):
                self.llm_engine.abort_request(group.request_id)

            self.llm_engine.scheduler.free_finished_seq_groups()

            assert not self.llm_engine.has_unfinished_requests()

        def cleanup(self):
            self.prompt = None
            self.request_counter = Counter()
            self.particle_metadata = VLLMParticleMetadata()
            self.free_unfinished_requests()
else:
    warnings.warn(
        'vllm is not installed. GPU-accelerated batch inference will not be available. Install vllm for improved performance.',
        ImportWarning,
    )
    BatchVLLM = None
