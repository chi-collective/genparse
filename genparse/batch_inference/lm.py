import copy
import torch
import warnings
import numpy as np
from arsenal.maths import logsumexp
from collections import defaultdict
from contextlib import contextmanager

import vllm
from vllm.sequence import (
    Logprob,
    SamplerOutput,
    SequenceOutput,
    CompletionSequenceGroupOutput,
)
from vllm.utils import Counter
from vllm import SamplingParams
from vllm.sequence import SequenceStatus
from vllm.sequence import ExecuteModelRequest
from vllm.engine.output_processor.util import create_output_by_sequence_group

from genparse.lm import VirtualTokenizedLLM
from genparse.util import load_model_by_name


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
        if isinstance(prompt, str):
            self.prompt = self.llm.encode_prompt(prompt)
        elif isinstance(prompt, tuple):
            self.prompt = prompt
        else:
            raise ValueError('prompt must be a string or a tuple of token ids')

    def batch_next_token_logprobs(self, particles, *args, **kwargs):
        logprobs = []
        particle_id_to_logprob_idx = {}
        for i, p in enumerate(particles):
            if not p.done:
                logprob = self.llm.logp_next(self.prompt + p.context)
                logprobs.append(logprob._p)
                particle_id_to_logprob_idx[i] = (0, len(logprobs) - 1)

        logprobs_by_seq_group = np.array([logprobs])  # XXX assume single prompt

        return logprobs_by_seq_group, particle_id_to_logprob_idx

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


class BatchVLLM(vllm.LLM):
    """
    Batch LM sampling and scoring with VLLM.

    This class provides an interface for batched next token logprob and scoring computations with the VLLM engine.

    Attributes:
        llm (VirtualTokenizedLLM): The VirtualTokenizedLLM instance.
        llm_engine (VLLMEngine): The VLLM engine.
        eos (str): The end-of-sequence token.
        eos_token_id (int): The end-of-sequence token id.
        request_counter (Counter): A counter for tracking the number of requests.
        particle_metadata (VLLMParticleMetadata):
            Metadata for tracking VLLM sequence information associated with the particle state.
        prompt (str): The LM prompt.
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

        self.prompt = None
        self.seq_group_metadata_list = None
        self.scheduler_outputs = None

    @classmethod
    def from_name(cls, model_name):
        return cls(llm=VirtualTokenizedLLM.from_name(model_name))

    def set_prompt(self, prompt):
        self.prompt = prompt

    def _make_initial_request(self, prompt):
        if prompt is None:
            raise ValueError(
                'Initial request requires a prompt. '
                'Use the `set_prompt` method to set the prompt.'
            )
        # clear previous requests
        self.reset_scheduler()
        self.seq_group_metadata_list = None
        self.scheduler_outputs = None
        request_id = self._validate_and_add_request(prompt)

        return request_id

    def _validate_and_add_request(self, prompt):
        if not isinstance(prompt, str):
            raise ValueError('prompt must be a string.')  # TODO: support token ids
        request_id = str(next(self.request_counter))
        # [0] since we are provided a single prompt as input
        inputs = self._convert_v1_inputs(prompts=[prompt], prompt_token_ids=None)[0]
        self.llm_engine.add_request(
            request_id=request_id,
            inputs=inputs,
            params=SamplingParams(
                max_tokens=np.inf, stop_token_ids=[], stop=None, ignore_eos=True
            ),
            lora_request=None,
        )
        return request_id

    def step_engine(self):
        """
        Run a single step of the VLLM engine to obtain the next token logprobs.

        Calls the scheduler and executes the model request to obtain the next token logprobs.
        """
        seq_group_metadata_list, scheduler_outputs = self.llm_engine.scheduler.schedule()

        # This statement will error if no initial requests are made, or if the scheduler for some
        # reason does not schedule the sequences with `waiting` status. The later occurs
        # when the block manager assigns an AllocStatus.LATER to the sequence group. This
        # can happen if the block manager is unable to allocate the necessary blocks for the sequence.
        # Calling `reset_scheduler` will clear the block manager and free up the space, which we do upon initial requests.
        assert (
            len(seq_group_metadata_list) > 0
        ), 'Scheduler did not schedule any sequences'

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

        assert (
            len(logprobs_by_step) == 1
        ), 'We should only be decoding a single step (logprobs)'
        assert (
            len(sequence_ids_by_step) == 1
        ), 'We should only be decoding a single step (sequence ids)'

        logprobs_by_seq_group = logprobs_by_step[0]  # single step
        sequence_ids_by_seq_group = sequence_ids_by_step[0]  # single step

        assert len(logprobs_by_seq_group) == len(
            seq_group_metadata_list
        ), 'Logprobs should be provided for each sequence group'
        assert len(sequence_ids_by_seq_group) == len(
            seq_group_metadata_list
        ), 'Sequence ids should be provided for each sequence group'

        return (
            logprobs_by_seq_group,
            sequence_ids_by_seq_group,
            seq_group_metadata_list,
            scheduler_outputs,
        )

    def extend_sequences_from_particles(
        self, particles, seq_group_metadata_list, scheduler_outputs
    ):
        if seq_group_metadata_list is None:
            # TODO: create the sequence group from the particles for this case
            # We can probably explore an implementation which creates a sequence group from the
            # particle prefixes at each time-step, as opposed to propagating the sequence group
            # across different calls to batch_next_token_logprobs. This would make the implementation
            # less dependent on maintaining the state of the vllm engine. Unclear whether this would
            # be as fast though. Maybe it's a fallback for cases in which the particles given as input
            # to batch_next_token_logprobs are not associated with any running sequences.
            raise ValueError(
                'No sequence group metadata associated with input particles.'
            )

        if scheduler_outputs is None:
            raise ValueError('No scheduler outputs associated with input particles.')

        context_to_particle_extensions = defaultdict(list)
        for particle_idx, particle in enumerate(particles):
            if not particle.done:
                context_to_particle_extensions[tuple(particle.context_ids[:-1])].append(
                    (particle_idx, particle.context_ids[-1])
                )

        assert (
            len(seq_group_metadata_list) == 1
        ), 'Only a single sequence group is supported for next token logprobs'

        extended_particles = []
        seq_outputs_by_seq_group = []
        for seq_group_idx, seq_group_metadata in enumerate(seq_group_metadata_list):
            sequence_outputs = []
            for seq_id, seq_data in seq_group_metadata.seq_data.items():
                extensions = context_to_particle_extensions[
                    tuple(seq_data.output_token_ids)
                ]

                if not extensions:
                    seq_data.status = SequenceStatus.FINISHED_STOPPED
                    continue

                unique_token_ids = []
                for particle_idx, sampled_token_id in extensions:
                    extended_particles.append(particle_idx)
                    if sampled_token_id not in unique_token_ids:
                        unique_token_ids.append(sampled_token_id)
                        sequence_outputs.append(
                            SequenceOutput(
                                parent_seq_id=seq_id,
                                output_token=sampled_token_id,
                                logprobs={sampled_token_id: Logprob(logprob=0)},
                            )
                        )
            seq_outputs_by_seq_group.append(sequence_outputs)
        # This will fail if the vllm engine does not contain the sequence groups which generated
        # the particle state. Eventually we may want detect when this is the case, and initialize
        # the approapriate vllm sequences (related to the TODO at the top of this function).
        assert all(
            p.done or (p_idx in extended_particles) for p_idx, p in enumerate(particles)
        ), 'All unfinished particles should be associated with a running sequence'

        self.apply_sequence_outputs(
            seq_outputs_by_seq_group=seq_outputs_by_seq_group,
            seq_group_metadata_list=seq_group_metadata_list,
            scheduler_outputs=scheduler_outputs,
        )

    def apply_sequence_outputs(
        self, seq_outputs_by_seq_group, seq_group_metadata_list, scheduler_outputs
    ):
        assert len(seq_group_metadata_list) == len(
            seq_outputs_by_seq_group
        ), 'Outputs must be provided for each running sequence group'

        sequence_outputs = [  # single step
            SamplerOutput(
                outputs=[
                    CompletionSequenceGroupOutput(
                        samples=sequence_outputs, prompt_logprobs=None
                    )  # for each sequence group
                    for sequence_outputs in seq_outputs_by_seq_group
                ]
            )
        ]

        self.llm_engine._process_model_outputs(
            sequence_outputs,
            scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups,
            seq_group_metadata_list,
        )

        self.llm_engine.scheduler.free_finished_seq_groups()

    def batch_next_token_logprobs(self, particles, is_initial=False):
        """Take a single VLLM step to compute logprobs for the next token"""
        if is_initial:
            # initialize the VLLM state with a sequence group for the prompt
            request_id = self._make_initial_request(self.prompt)
            for p in particles:
                p.prompt = request_id
        else:
            # extend the VLLM sequences with the particle extensions
            self.extend_sequences_from_particles(
                particles, self.seq_group_metadata_list, self.scheduler_outputs
            )

        (
            logprobs_by_seq_group,
            sequence_ids_by_seq_group,
            seq_group_metadata_list,
            scheduler_outputs,
        ) = self.step_engine()
        # Save scheduler outputs and metadata for subsequent batch steps (ugly)
        # We need to save this state to extend sequences from extended particles at the next call to batch_next_token_logprobs.
        # Perhaps a more elegant solution would be to use a datastructure which stored the information necc
        # to reconstruct these objects based on the (seq ids for the) particles and the state of the vllm engine.
        self.seq_group_metadata_list = seq_group_metadata_list
        self.scheduler_outputs = scheduler_outputs

        # map particles to sequences according to the particle context ids
        # maybe TODO: instead of having to compute the sequence ids for each context at each step,
        # store the sequence id in the particles themselves.

        context_ids_to_seq_id = {}  # request_id -> seq_context_ids -> [seq_group_idx, seq_id]
        for seq_group_idx, seq_group_metadata in enumerate(seq_group_metadata_list):
            context_ids_to_seq_id[seq_group_metadata.request_id] = {}
            for seq_id, seq_data in seq_group_metadata.seq_data.items():
                seq_context_ids = tuple(seq_data.output_token_ids)
                # duplicate sequences shouldn't exist if the state of the vllm_engine isn't modified in between steps,
                # and if previous unfinished requests are cleared (which we do upon inital requests)
                assert (
                    seq_context_ids
                    not in context_ids_to_seq_id[seq_group_metadata.request_id]
                ), 'Detected duplicate vllm sequence'
                context_ids_to_seq_id[seq_group_metadata.request_id][seq_context_ids] = [
                    seq_group_idx,
                    seq_id,
                ]

        particle_idx_to_logprob_idx = []  # particle_idx -> [seq_group_idx, seq_idx]
        for particle in particles:
            assert particle.prompt is not None, 'All particles should have a prompt id'
            if particle.done:
                particle_idx_to_logprob_idx.append([None, None])
            else:
                seq_group_seq_ids = context_ids_to_seq_id.get(particle.prompt, None)
                assert (
                    seq_group_seq_ids is not None
                ), 'All unfinished particles should be associated with a running sequence group'
                seq_group_idx, seq_id = seq_group_seq_ids.get(
                    tuple(particle.context_ids), (None, None)
                )
                assert (seq_group_idx is not None) and (
                    seq_id is not None
                ), 'All unfinished particles should be associated with a running sequence'
                particle_idx_to_logprob_idx.append(
                    (
                        seq_group_idx,
                        sequence_ids_by_seq_group[seq_group_idx].index(seq_id),
                    )
                )

        return np.array(logprobs_by_seq_group), particle_idx_to_logprob_idx

    @contextmanager
    def swap_running_sequence_groups(self):
        """
        This context manager is used to "pause" the currently running sequence group in order
        to run another. This is particularly useful to score sequences under the LM
        in the middle of SMC inference.
        """
        # This swap does not free the *blocks* associated with the sequence group.
        # A block is a contiguous chunk of memory which
        # Typically this is fine since there is enough space in the block manager
        # TODO: check if the block manager is full and recompute these requests if so.
        swapped_out_seq_groups = []
        while self.llm_engine.scheduler.running:
            seq_group = self.llm_engine.scheduler.running.pop()
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq.status = SequenceStatus.SWAPPED
            swapped_out_seq_groups.append(seq_group)

        try:
            yield
        finally:
            for seq_group in swapped_out_seq_groups:
                for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
                    seq.status = SequenceStatus.RUNNING
            self.llm_engine.scheduler.running.extend(swapped_out_seq_groups)

    def batch_score_sequences(self, prompts, token_ids, temperature=1):
        """Compute log p(token_ids | prompt) for each prompt."""
        with self.swap_running_sequence_groups():
            request_ids = [
                self._validate_and_add_request(prompt=prompt) for prompt in prompts
            ]

            # This could likely be made much faster by increasing the step size.
            # Instead of stepping the engine for each token, we could call it once
            # and have it compute the logprobs for len(token_ids) steps. It would also
            # remove much of this logic, and help us support SMC inference with step size greater than 1.

            logprobs = np.zeros(len(request_ids))
            for i, token_id in enumerate(token_ids):
                (
                    logprobs_by_seq_group,
                    sequence_ids_by_seq_group,
                    seq_group_metadata_list,
                    scheduler_outputs,
                ) = self.step_engine()

                seq_outputs_by_seq_group = []
                for seq_group_idx, seq_group_metadata in enumerate(
                    seq_group_metadata_list
                ):
                    prompt_idx = request_ids.index(seq_group_metadata.request_id)
                    seq_group_seq_ids = sequence_ids_by_seq_group[seq_group_idx]
                    assert (
                        len(seq_group_seq_ids) == 1
                    ), 'Multiple seqs not supported for scoring'
                    seq_id = seq_group_seq_ids[0]

                    token_logprob = self._temper_logprobs(
                        logprobs_by_seq_group[seq_group_idx][0], temperature
                    )[token_id]
                    logprobs[prompt_idx] += token_logprob

                    sequence_outputs = [
                        SequenceOutput(
                            parent_seq_id=seq_id,
                            output_token=token_id,
                            logprobs={token_id: Logprob(logprob=token_logprob)},
                        )
                    ]

                    seq_outputs_by_seq_group.append(sequence_outputs)

                self.apply_sequence_outputs(
                    seq_outputs_by_seq_group=seq_outputs_by_seq_group,
                    seq_group_metadata_list=seq_group_metadata_list,
                    scheduler_outputs=scheduler_outputs,
                )

            self.free_running_sequence_groups()

        return logprobs

    def _temper_logprobs(self, logprobs, temperature):
        if temperature != 1:
            logprobs = logprobs / temperature
            logprobs = logprobs - logsumexp(logprobs)
        return logprobs

    def free_running_sequence_groups(self):
        while self.llm_engine.scheduler.running:
            seq_group = self.llm_engine.scheduler.running.pop()
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # release the blocks associated with the sequence
                self.llm_engine.scheduler.free_seq(seq)
            self.llm_engine.abort_request(seq_group.request_id)

    def free_swapped_sequence_groups(self):
        while self.llm_engine.scheduler.swapped:
            seq_group = self.llm_engine.scheduler.swapped.pop()
            self.llm_engine.abort_request(seq_group.request_id)

    def free_waiting_sequence_groups(self):
        while self.llm_engine.scheduler.waiting:
            seq_group = self.llm_engine.scheduler.waiting.pop()
            self.llm_engine.abort_request(seq_group.request_id)

    def reset_scheduler(self):
        """Free all unfinished requests from the scheduler and reset block manager."""
        self.free_running_sequence_groups()
        self.free_waiting_sequence_groups()
        self.free_swapped_sequence_groups()
        self.llm_engine.scheduler.free_finished_seq_groups()
        self.llm_engine.scheduler.block_manager.reset()
        assert not self.llm_engine.has_unfinished_requests()

    def cleanup(self):
        self.prompt = None
        self.seq_group_metadata_list = None
        self.scheduler_outputs = None
        self.reset_scheduler()
