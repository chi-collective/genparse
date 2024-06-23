"""
Language model steering methods (VLLM compatible)
"""
from arsenal import colors, timers

import asyncio
import random
import warnings
import copy
from collections import defaultdict

import numpy as np
import torch
import transformers
from arsenal.maths import logsumexp, sample_dict

from hfppl import Model

from genparse import EOS
from genparse.vllm_inference import (
    TraceSWOR,
    importance_sampling,
    smc_standard,
    smc_standard_record,
    smc_steer,
    VLLMParticle
)
from genparse.lm import LM
from genparse.semiring import Float
from genparse.util import format_table, set_seed
from genparse.steer import ParticleApproximation


class VLLMWrapper:
    def __init__(self, llm, n_particles, guide, proposal, prompt, max_tokens, verbosity=0, timer=None):
        from vllm.sampling_params import SamplingParams
        from genparse.tokenization import decode_tokenizer_vocab

        super().__init__()
        # type: AsyncGreedilyTokenizedLLM
        self.llm = llm
        self.n_particles = n_particles

        """
            One VLLMWrapper is initialized for each prompt.
            All VLLMWrapper point to the same AsyncGreedilyTokenizedLLM 
            based on one VLLM instance (self.llm).
            We add the prompt in the VLLMWrapper constructor.
        """

        self.particles = {}

        self.token_to_id = {
            x: i for i, x in enumerate(decode_tokenizer_vocab(self.llm.tokenizer))
        }

        self.max_tokens = max_tokens
        self.guide = guide
        self.prompt = prompt

        self.proposal = proposal  # the original proposal

        self.verbosity = verbosity
        self.timer = timer

    def add_prompt(self, prompt):
        from vllm.sampling_params import SamplingParams

        if isinstance(prompt, str):
            request_id = str(
                next(copy.deepcopy(self.llm._model.request_counter)))
            self.particles[request_id] = [
                VLLMParticle(
                    prompt=prompt, max_tokens=self.max_tokens, proposal=copy.deepcopy(
                        self.proposal)
                ) for _ in range(self.n_particles)
            ]

            self.llm._model._validate_and_add_requests(
                inputs=self.llm._model._convert_v1_inputs(
                    prompts=prompt, prompt_token_ids=None),
                # using default params because we only rely on logits
                params=SamplingParams(max_tokens=self.max_tokens,
                                      stop_token_ids=[self.llm.eos]),
                lora_request=None
            )
            return request_id
        elif isinstance(prompt, list):
            request_ids = []
            for p in prompt:
                request_ids.append(self.add_prompt(p))
            return request_ids

    async def prepare_logprobs(self):
        from vllm.engine.output_processor.util import create_output_by_sequence_group
        from vllm.sequence import (
            CompletionSequenceGroupOutput,
            SequenceOutput,
            ExecuteModelRequest,
            SamplerOutput,
            Logprob
        )

        seq_group_metadata_list, scheduler_outputs = (
            self.llm._model.llm_engine.scheduler.schedule()
        )

        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
            num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
            running_queue_size=scheduler_outputs.running_queue_size,
        )
        # logprobs, seq_ids are
        # list of torch.Tensor each with shape [n_sample, vocab_size]
        res = self.llm._model.llm_engine.model_executor.execute_model(
            execute_model_req=execute_model_req)
        logprobs, seq_ids = zip(*res)

        next_logprobs_by_sequence_group = create_output_by_sequence_group(
            logprobs, num_seq_groups=len(scheduler_outputs.scheduled_seq_groups))
        next_seq_ids_by_sequence_group = create_output_by_sequence_group(
            seq_ids, num_seq_groups=len(scheduler_outputs.scheduled_seq_groups))

        return scheduler_outputs, seq_group_metadata_list, next_logprobs_by_sequence_group, next_seq_ids_by_sequence_group

    async def sample_next_token(self, scheduler_outputs, seq_group_metadata_list, next_logprobs_by_sequence_group, next_seq_ids_by_sequence_group):

        from vllm.sequence import (
            CompletionSequenceGroupOutput,
            SequenceOutput,
            ExecuteModelRequest,
            SamplerOutput,
            Logprob
        )

        # sample next token with llm and guide

        results = []
        for seq_group_md, seq_group_next_logprob, seq_group_ids in zip(seq_group_metadata_list, next_logprobs_by_sequence_group, next_seq_ids_by_sequence_group):

            group_results = []
            assert len(
                seq_group_next_logprob) == 1, "We are using one-step decoding."
            # take the only step
            seq_group_next_logprob = seq_group_next_logprob[0]
            seq_group_ids = seq_group_ids[0]

            request_id = seq_group_md.request_id
            is_prompt = seq_group_md.is_prompt
            # running parent_ids
            parent_id_to_result_id = {
                parent_id: i for i, parent_id in enumerate(seq_group_ids)}

            output_ids_to_seq_ids = {
                tuple(x.output_token_ids): k for k, x in seq_group_md.seq_data.items()
            }

            for par_id, particle in enumerate(self.particles[request_id]):
                if is_prompt:
                    # all particles are forked from one prompt and are equivalent
                    # so we don't need a parent_id for them in this case
                    assert seq_group_next_logprob.size(
                        0) == 1, (seq_group_next_logprob.size(), len(self.particles[request_id]))
                    logp = seq_group_next_logprob[0]
                    parent_id = seq_group_ids[0]
                else:
                    # particles starting to diverge
                    particle_key = particle.context_ids_tuple()
                    if particle_key not in output_ids_to_seq_ids:
                        # this sequence is finished, but particle is still alive
                        particle.finish()
                        continue

                    particle.parent_id = output_ids_to_seq_ids[particle_key]
                    parent_id = particle.parent_id

                    if parent_id not in seq_group_ids or particle.finished:
                        # this sequence is finished, but particle is still alive
                        particle.finish()
                        continue
                    result_id = parent_id_to_result_id[parent_id]
                    logp = seq_group_next_logprob[result_id]

                (token, _, weight_update) = await particle.proposal.sample_next_token(
                    prompt=self.prompt,
                    context=''.join(particle.context),
                    p_llm=await self.llm.p_next(_logp=logp)
                )
                token_id = self.token_to_id.get(
                    token, self.llm._model.eos_token_id)
                particle.context.append(token)
                particle.context_ids.append(token_id)
                particle.weight += np.log(weight_update)
                particle.max_tokens -= 1

                group_results.append(
                    (token_id, weight_update, parent_id, par_id))

                if self.verbosity > 1:
                    print('particle, token, token_id', i, token, token_id)
                    print(
                        f"`{token} {token_id}` : {''.join(particle.context)} : {particle.weight}")

            results.append(group_results)

        return results

    def prepare_particles_for_next_step(
        self,
        scheduler_outputs,
        seq_group_metadata_list,
        results,
        repeats
    ):
        from vllm.sequence import (
            CompletionSequenceGroupOutput,
            SequenceOutput,
            ExecuteModelRequest,
            SamplerOutput,
            Logprob,
            SequenceStatus
        )
        # fork new particles and kill old ones
        # by repeating samples in processed_output or removing

        token_ids, weight_updates, parent_ids, particle_ids = [], [], [], []
        for group_results in results:
            token_ids.append([token_id for token_id, _, _, _ in group_results])
            weight_updates.append(
                [weight_update for _, weight_update, _, _ in group_results])
            parent_ids.append(
                [parent_id for _, _, parent_id, _ in group_results])

        processed_output = []

        for scheduled_seq_group, seq_group_md, next_token_ids, weight_update, group_parent_ids in zip(
            scheduler_outputs.scheduled_seq_groups, seq_group_metadata_list, token_ids, weight_updates, parent_ids
        ):
            request_id = seq_group_md.request_id
            seq_outputs: List[SequenceOutput] = []

            for parent_id, next_token_id, logprobs in zip(group_parent_ids, next_token_ids, weight_update):

                sample = SequenceOutput(
                    # parent_id is the id that the current continuation is based on
                    parent_seq_id=parent_id,
                    output_token=next_token_id,
                    logprobs={
                        next_token_id: Logprob(logprob=np.log(logprobs))
                    },
                )
                seq_outputs.append(sample)

            processed_output.append(
                SamplerOutput(
                    outputs=[
                        CompletionSequenceGroupOutput(
                            samples=seq_outputs,
                            prompt_logprobs=None,
                        )
                    ]
                )
            )

        return processed_output

    def process_model_outputs(self, scheduler_outputs, seq_group_metadata_list, processed_output):

        # Post-processing. Do after sample is chosen.
        # Feed output list to vllm engine scheduler and prepare for next step
        self.llm._model.llm_engine._process_model_outputs(
            processed_output,
            scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups,
            seq_group_metadata_list,
        )

        # Log stats.
        self.llm._model.llm_engine.do_log_stats(
            scheduler_outputs, processed_output)

        return

    async def update(self):
        # prepare logprobs
        with self.timer['llm'](t=len("".join(list(self.particles.values())[0][0].context))):
            scheduler_outputs, seq_group_metadata_list, next_logprobs_by_sequence_group, next_seq_ids_by_sequence_group = await self.prepare_logprobs()
        # sampling
        with self.timer['cfg+trie'](t=len("".join(list(self.particles.values())[0][0].context))):
            results = await self.sample_next_token(scheduler_outputs, seq_group_metadata_list, next_logprobs_by_sequence_group, next_seq_ids_by_sequence_group)

        return scheduler_outputs, seq_group_metadata_list, results

    def postprocess(self, scheduler_outputs, seq_group_metadata_list, results, repeats):

        processed_output = self.prepare_particles_for_next_step(
            scheduler_outputs, seq_group_metadata_list, results, repeats)
        # post processing: update scheduler states
        self.process_model_outputs(
            scheduler_outputs, seq_group_metadata_list, processed_output)

        return

    async def step(self):
        scheduler_outputs, seq_group_metadata_list, results = await self.update()

        repeats = defaultdict(lambda: 1)
        self.postprocess(scheduler_outputs,
                         seq_group_metadata_list, results, repeats)

        return


class VLLMSampler:
    def __init__(self, llm, guide):
        """
        Args:
            llm (vllm.VLLM)
            guide (LM)
        Returns:
            particle_approximation (ParticleApproximation)
            record (dict | NoneType): information about the run
        """
        self.llm = llm
        self.guide = guide
        self.timer = timers()

    def run_inference(
        self,
        prompt,
        proposal,
        method,
        n_particles,
        n_beam=None,
        max_tokens=float('inf'),
        verbosity=0,
        return_record=False,
        seed=None,
    ):
        if seed is not None:
            set_seed(seed)

        model = VLLMWrapper(
            llm=self.llm,
            n_particles=n_particles,
            guide=self.guide,
            prompt=prompt,
            proposal=proposal,
            max_tokens=max_tokens,
            verbosity=verbosity,
            timer=self.timer
        )

        record = None
        if method == 'smc-steer':
            assert n_beam is not None
            if return_record:
                raise Warning('Record not yet implemented for smc-steer')
            particles = asyncio.run(
                smc_steer(model, n_particles=n_particles, n_beam=n_beam)
            )

        elif method == 'smc-standard':
            if return_record:
                particles, record = asyncio.run(
                    smc_standard_record(
                        model, n_particles=n_particles, return_record=return_record
                    )
                )
            else:
                particles = asyncio.run(smc_standard(
                    model, n_particles=n_particles))

        elif method == 'importance-sampling':
            particles = asyncio.run(importance_sampling(
                model, n_particles=n_particles))

        else:
            raise ValueError(f'Unknown inference method: {method}.')

        return ParticleApproximation(particles), record
