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
        self.prompt = self.llm.encode_prompt(prompt)

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
        request_id = str(next(self.request_counter))
        assert isinstance(prompt, str), 'Prompt must be a string.'
        self.llm_engine.add_request(
            request_id=request_id,
            inputs=self._convert_v1_inputs(prompts=prompt, prompt_token_ids=None)[0],
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
        # Calling `reset_scheduler` will clear the block manager and free up the space.
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

        logprobs_by_seq_group = logprobs_by_step[0]
        sequence_ids_by_seq_group = sequence_ids_by_step[0]

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
                # we can ignore finished particles since we do not need to make vllm requests for them
                context_to_particle_extensions[tuple(particle.context_ids[:-1])].append(
                    (particle_idx, particle.context_ids[-1])
                )

        assert (
            len(seq_group_metadata_list) == 1
        ), 'Only a single sequence group is supported for next token logprobs'

        all_sequence_outputs = []
        extended_particles = []
        for seq_group_idx, seq_group_metadata in enumerate(seq_group_metadata_list):
            sequence_outputs = []
            for seq_id, seq_data in seq_group_metadata.seq_data.items():
                extensions = context_to_particle_extensions[
                    tuple(seq_data.output_token_ids)
                ]

                if not extensions:
                    seq_data.status = SequenceStatus.FINISHED_STOPPED
                    # XXX free blocks here?
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
            all_sequence_outputs.append(sequence_outputs)
        # This will fail if the vllm engine does not contain the sequence groups which generated
        # the particle state. Eventually we may want detect when this is the case, and initialize
        # the approapriate vllm sequences (related to the TODO at the top of this function).
        assert all(
            p.done or (p_idx in extended_particles) for p_idx, p in enumerate(particles)
        ), 'All unfinished particles should be associated with a running sequence'

        self.apply_sequence_outputs(
            all_sequence_outputs=all_sequence_outputs,
            seq_group_metadata_list=seq_group_metadata_list,
            scheduler_outputs=scheduler_outputs,
        )

    def apply_sequence_outputs(
        self, all_sequence_outputs, seq_group_metadata_list, scheduler_outputs
    ):
        assert len(seq_group_metadata_list) == len(
            all_sequence_outputs
        ), 'Outputs must be provided for each running sequence group'

        sequence_outputs = [
            SamplerOutput(
                outputs=[
                    CompletionSequenceGroupOutput(
                        samples=sequence_outputs, prompt_logprobs=None
                    )
                ]
            )
            for sequence_outputs in all_sequence_outputs
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
            request_id = self._make_initial_request(self.prompt)
            for p in particles:
                p._replace(prompt=request_id)
        else:
            self.extend_sequences_from_particles(
                particles, self.seq_group_metadata_list, self.scheduler_outputs
            )

        (
            logprobs_by_seq_group,
            sequence_ids_by_seq_group,
            seq_group_metadata_list,
            scheduler_outputs,
        ) = self.step_engine()
        # Save scheduler outputs and md for subsequent batch steps (ugly)
        # We need to save this state to extend sequences from extended particles at the next call to batch_next_token_logprobs.
        # Perhaps a more elegant solution would be to use a datastructure which stored the information necc
        # to reconstruct these objects based on the (seq ids for the) particles and the state of the vllm engine.
        self.seq_group_metadata_list = seq_group_metadata_list
        self.scheduler_outputs = scheduler_outputs

        # request_id -> seq context_ids -> [seq_group_idx, seq_id]
        context_ids_to_seq_id = {}
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

        # particle_idx -> [seq_group_idx, seq_id]
        particle_idx_to_logprob_idx = []
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
                assert (
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
        # This swap does not free the blocks associated with the sequence group
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

    def batch_score_sequences(self, prompt, token_ids, masks=None, temperature=1):
        """
        Compute log p(token_ids | prompt) for a batch of token_ids.

        Args:
            prompt (str): The prompt.
            token_ids (list): A list of lists of token ids.
            masks (list): A list of lists of boolean masks, one for each token id list.
                mask[i,j] = False does not score token_ids[i,j]. Defaults to scoring all tokens.

        Returns:
            logprobs (np.ndarray): The log probabilities.
        """
        max_length = max(len(l) for l in token_ids)
        token_ids = np.array([x + [-1] * (max_length - len(x)) for x in token_ids])

        if masks is None:
            masks = np.ones(token_ids.shape, dtype=bool)
        else:
            masks = np.array([x + [0] * (max_length - len(x)) for x in masks], dtype=bool)

        assert (
            token_ids.shape == masks.shape
        ), 'Token ids and masks must have the same shape'

        with self.swap_running_sequence_groups():
            self._validate_and_add_request(prompt=prompt)
            logprobs = self._batch_compute_logprobs(
                token_ids=token_ids, masks=masks, temperature=temperature
            )

        return logprobs

    def _batch_compute_logprobs(self, token_ids, masks, temperature=1):
        # It is possible to compute the logprobs for sequences by running a `generate` request
        # on the vllm LLM with prompt_logprobs=0 and max_tokens=1.
        # However, in practice I've found that this leads to a lot of memory leakage, which makes
        # it essentially unusable.
        token_logprobs = np.zeros(len(token_ids))
        for i, step_ids in enumerate(token_ids.T):
            (
                logprobs_by_seq_group,
                sequence_ids_by_seq_group,
                seq_group_metadata_list,
                scheduler_outputs,
            ) = self.step_engine()
            all_sequence_outputs = []
            for seq_group_idx, seq_group_metadata in enumerate(seq_group_metadata_list):
                sequence_outputs = []
                seq_group_logprobs = logprobs_by_seq_group[seq_group_idx]
                seq_group_seq_ids = sequence_ids_by_seq_group[seq_group_idx]
                for seq_id, seq_data in seq_group_metadata.seq_data.items():
                    next_token_logprobs = seq_group_logprobs[
                        seq_group_seq_ids.index(seq_id)
                    ]

                    if temperature != 1:
                        next_token_logprobs = next_token_logprobs / temperature
                        next_token_logprobs = next_token_logprobs - logsumexp(
                            next_token_logprobs
                        )

                    token_list_ids = np.where(
                        np.all(token_ids[:, :i] == seq_data.output_token_ids, axis=1)
                    )[0]

                    is_finished = True
                    unique_token_ids = []
                    for token_list_id in token_list_ids:
                        token_id = step_ids[token_list_id]

                        if token_id == -1:
                            continue

                        is_finished = False
                        if token_id not in unique_token_ids:
                            unique_token_ids.append(token_id)

                            if masks[token_list_id, i]:
                                token_logprob = next_token_logprobs[token_id]
                                token_logprobs[token_list_id] += token_logprob
                            else:
                                token_logprob = 0

                            sequence_outputs.append(
                                SequenceOutput(
                                    parent_seq_id=seq_id,
                                    output_token=token_id,
                                    logprobs={token_id: Logprob(logprob=token_logprob)},
                                )
                            )

                    if is_finished:
                        seq_data.status = SequenceStatus.FINISHED_STOPPED
                        # XXX Free blocks here?

                all_sequence_outputs.append(sequence_outputs)

            self.apply_sequence_outputs(
                all_sequence_outputs=all_sequence_outputs,
                seq_group_metadata_list=seq_group_metadata_list,
                scheduler_outputs=scheduler_outputs,
            )

        self.free_running_sequence_groups()

        return token_logprobs

    def batch_score_marginal_sequence(
        self, prompts, marginal_eos_token_id, token_ids, n_marginal, temperature=1
    ):
        raise NotImplementedError('This method is not yet implemented')
        # XXX: very inefficient

        # for each prompt, sample n_marginal completions, then score token_ids

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
