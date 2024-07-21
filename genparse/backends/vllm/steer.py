"""
Language model steering methods (VLLM compatible)
"""

from arsenal import timers
import asyncio
import copy
import numpy as np
from collections import defaultdict

from arsenal.maths import logsumexp
from arsenal import colors

from genparse.record import SMCRecord
from genparse.util import set_seed
from genparse.steer import ParticleApproximation
from genparse.tokenization import decode_tokenizer_vocab

try:
    from vllm.sampling_params import SamplingParams
    from vllm.engine.output_processor.util import create_output_by_sequence_group
    from vllm.sequence import ExecuteModelRequest
    from vllm.sequence import (
        CompletionSequenceGroupOutput,
        SequenceOutput,
        SamplerOutput,
        Logprob,
    )
except ImportError:
    pass


class VLLMWrapper:
    def __init__(
        self,
        llm,
        n_particles,
        guide,
        proposal,
        prompt,
        max_tokens,
        verbosity=0,
        timer=None,
    ):
        self.llm = llm
        self.n_particles = n_particles

        # One VLLMWrapper is initialized for each prompt.
        # All VLLMWrapper point to the same TokenizedLLM
        # based on one VLLM instance (self.llm).
        # We add the prompt in the VLLMWrapper constructor.

        self.particles = {}

        self.token_to_id = {
            x: i for i, x in enumerate(decode_tokenizer_vocab(self.llm.tokenizer))
        }

        self.max_tokens = max_tokens
        self.guide = guide
        self.prompt = prompt
        if not prompt.startswith(self.llm.tokenizer.bos_token):
            self.prompt = self.llm.tokenizer.bos_token + prompt

        self.proposal = proposal  # the original proposal

        self.verbosity = verbosity
        self.timer = timer

    def add_prompt(self, prompt):
        if isinstance(prompt, str):
            request_id = str(next(copy.deepcopy(self.llm._model.request_counter)))
            self.particles[request_id] = [
                VLLMParticle(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                )
                for _ in range(self.n_particles)
            ]

            self.llm._model._validate_and_add_requests(
                inputs=self.llm._model._convert_v1_inputs(
                    prompts=prompt, prompt_token_ids=None
                ),
                # using default params because we only rely on logits
                params=SamplingParams(
                    max_tokens=self.max_tokens, stop_token_ids=[self.llm.eos]
                ),
                lora_request=None,
            )
            return request_id
        elif isinstance(prompt, list):
            request_ids = []
            for p in prompt:
                request_ids.append(self.add_prompt(p))
            return request_ids

    async def prepare_logprobs(self):
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
            execute_model_req=execute_model_req
        )
        logprobs, seq_ids = zip(*res)

        next_logprobs_by_sequence_group = create_output_by_sequence_group(
            logprobs, num_seq_groups=len(scheduler_outputs.scheduled_seq_groups)
        )
        next_seq_ids_by_sequence_group = create_output_by_sequence_group(
            seq_ids, num_seq_groups=len(scheduler_outputs.scheduled_seq_groups)
        )

        return (
            scheduler_outputs,
            seq_group_metadata_list,
            next_logprobs_by_sequence_group,
            next_seq_ids_by_sequence_group,
        )

    async def sample_next_token(
        self,
        scheduler_outputs,
        seq_group_metadata_list,
        next_logprobs_by_sequence_group,
        next_seq_ids_by_sequence_group,
    ):
        # sample next token with llm and guide

        results = []
        for seq_group_md, seq_group_next_logprob, seq_group_ids in zip(
            seq_group_metadata_list,
            next_logprobs_by_sequence_group,
            next_seq_ids_by_sequence_group,
        ):
            group_results = []
            assert len(seq_group_next_logprob) == 1, 'We are using one-step decoding.'
            # take the only step
            seq_group_next_logprob = seq_group_next_logprob[0]
            seq_group_ids = seq_group_ids[0]

            request_id = seq_group_md.request_id
            is_prompt = seq_group_md.is_prompt
            # running parent_ids
            parent_id_to_result_id = {
                parent_id: i for i, parent_id in enumerate(seq_group_ids)
            }

            output_ids_to_seq_ids = {
                tuple(x.output_token_ids): k for k, x in seq_group_md.seq_data.items()
            }

            for par_id, particle in enumerate(self.particles[request_id]):
                if is_prompt:
                    # all particles are forked from one prompt and are equivalent
                    # so we don't need a parent_id for them in this case
                    assert seq_group_next_logprob.size(0) == 1, (
                        seq_group_next_logprob.size(),
                        len(self.particles[request_id]),
                    )
                    logp = seq_group_next_logprob[0]
                    parent_id = seq_group_ids[0]
                else:
                    # particles starting to diverge
                    particle_key = particle.context_ids_tuple()
                    if particle_key not in output_ids_to_seq_ids:
                        # this sequence is finished, but particle is still alive
                        particle.finished = True
                        continue

                    particle.parent_id = output_ids_to_seq_ids[particle_key]
                    parent_id = particle.parent_id

                    if parent_id not in seq_group_ids or particle.finished:
                        # this sequence is finished, but particle is still alive
                        particle.finished = True
                        continue
                    result_id = parent_id_to_result_id[parent_id]
                    logp = seq_group_next_logprob[result_id]

                (token, _, weight_update) = await self.proposal.sample_next_token(
                    prompt=self.prompt,
                    context=tuple(particle.context),
                    p_llm=await self.llm.p_next_async(context=None, _logp=logp),
                )
                token_id = self.token_to_id.get(token, self.llm._model.eos_token_id)
                particle.context.append(token)
                particle.context_ids.append(token_id)
                particle.weight += np.log(weight_update)
                particle.max_tokens -= 1

                group_results.append((token_id, weight_update, parent_id, par_id))

                if self.verbosity > 1:
                    print('token, token_id', token, token_id)
                    print(
                        f"`{token} {token_id}` : {''.join(particle.context)} : {particle.weight}"
                    )

            results.append(group_results)

        return results

    def prepare_particles_for_next_step(
        self, scheduler_outputs, seq_group_metadata_list, results, repeats
    ):
        # fork new particles and kill old ones
        # by repeating samples in processed_output or removing

        token_ids, weight_updates, parent_ids = [], [], []
        for group_results in results:
            token_ids.append([token_id for token_id, _, _, _ in group_results])
            weight_updates.append(
                [weight_update for _, weight_update, _, _ in group_results]
            )
            parent_ids.append([parent_id for _, _, parent_id, _ in group_results])

        processed_output = []

        for (
            scheduled_seq_group,
            seq_group_md,
            next_token_ids,
            weight_update,
            group_parent_ids,
        ) in zip(
            scheduler_outputs.scheduled_seq_groups,
            seq_group_metadata_list,
            token_ids,
            weight_updates,
            parent_ids,
        ):
            seq_outputs = []

            for parent_id, next_token_id, logprobs in zip(
                group_parent_ids, next_token_ids, weight_update
            ):
                sample = SequenceOutput(
                    # parent_id is the id that the current continuation is based on
                    parent_seq_id=parent_id,
                    output_token=next_token_id,
                    logprobs={next_token_id: Logprob(logprob=np.log(logprobs))},
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

    def process_model_outputs(
        self, scheduler_outputs, seq_group_metadata_list, processed_output
    ):
        # Post-processing. Do after sample is chosen.
        # Feed output list to vllm engine scheduler and prepare for next step
        self.llm._model.llm_engine._process_model_outputs(
            processed_output,
            scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups,
            seq_group_metadata_list,
        )

        # Log stats.
        self.llm._model.llm_engine.do_log_stats(scheduler_outputs, processed_output)

    async def update(self):
        # prepare logprobs
        with self.timer['llm'](
            t=len(''.join(list(self.particles.values())[0][0].context))
        ):
            (
                scheduler_outputs,
                seq_group_metadata_list,
                next_logprobs_by_sequence_group,
                next_seq_ids_by_sequence_group,
            ) = await self.prepare_logprobs()
        # sampling
        with self.timer['cfg+trie'](
            t=len(''.join(list(self.particles.values())[0][0].context))
        ):
            results = await self.sample_next_token(
                scheduler_outputs,
                seq_group_metadata_list,
                next_logprobs_by_sequence_group,
                next_seq_ids_by_sequence_group,
            )

        return scheduler_outputs, seq_group_metadata_list, results

    def postprocess(self, scheduler_outputs, seq_group_metadata_list, results, repeats):
        processed_output = self.prepare_particles_for_next_step(
            scheduler_outputs, seq_group_metadata_list, results, repeats
        )
        # post processing: update scheduler states
        self.process_model_outputs(
            scheduler_outputs, seq_group_metadata_list, processed_output
        )

    async def step(self):
        scheduler_outputs, seq_group_metadata_list, results = await self.update()

        repeats = defaultdict(lambda: 1)
        self.postprocess(scheduler_outputs, seq_group_metadata_list, results, repeats)


class VLLMSampler:
    def __init__(self, llm, guide):
        """
        Args:
            llm (vllm.VLLM)
            guide (LM)
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
        ess_threshold=0.5,
        seed=None,
    ):
        """
        Returns:
            particle_approximation (ParticleApproximation)
        """
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
            timer=self.timer,
        )

        record = None

        if method == 'smc-standard':
            particles, record = asyncio.run(
                smc_standard(
                    model,
                    n_particles=n_particles,
                    return_record=return_record,
                    ess_threshold=ess_threshold,
                )
            )
        elif method == 'importance-sampling':
            particles, record = asyncio.run(
                smc_standard(
                    model,
                    n_particles=n_particles,
                    return_record=return_record,
                    ess_threshold=0,
                )
            )
        else:
            raise ValueError(f'unrecognized method name {method!r}')

        self.llm.clear_cache()
        self.guide.clear_cache()

        return ParticleApproximation(particles, record=record)


class VLLMParticle:
    def __init__(self, prompt, max_tokens):
        self.context = []
        self.context_ids = []
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.request_id = None
        self.parent_id = None
        self.finished = False
        self.weight = 0

    def __lt__(self, other):
        return self.weight < other.weight

    def context_ids_tuple(self):
        return tuple(self.context_ids)

    #    def __str__(self):
    #        return f'{" ".join(self.context)}'

    def __repr__(self):
        return (
            f'{self.weight:.2f}:\t'
            + colors.light.cyan % '['
            + (colors.light.cyan % '|').join(repr(y)[1:-1] for y in self.context)
            + colors.light.cyan % ']'
        )


# TODO: we should be able to use a single SMC implementation!
async def smc_standard(model, n_particles, ess_threshold=0.5, return_record=True):
    """
    Standard sequential Monte Carlo algorithm with multinomial resampling.

    Args:
        model (Model): The model to perform inference on.
        n_particles (int): Number of particles to execute concurrently.
        ess_threshold (float): Effective sample size below which resampling is triggered, given as a fraction of `n_particles`.

    Returns:
        particles: The completed particles after inference.
        record (SMCRecord): Information about inference run history.
    """
    verbosity = model.verbosity if hasattr(model, 'verbosity') else 0

    # Create n_particles copies of the model

    request_id = model.add_prompt(model.prompt)
    particles = model.particles[request_id]

    # Initialize record dict
    record = (
        SMCRecord(
            {
                'n_particles': n_particles,
                'ess_threshold': ess_threshold,
                'algorithm': 'smc_standard_record',
                'history': [],
            }
        )
        if return_record
        else None
    )

    if return_record or verbosity > 0:
        step_num = 1

    while model.llm._model.llm_engine.has_unfinished_requests():
        if return_record:
            step = {'step': step_num}

        # Step each particle

        await model.step()

        # Normalize weights
        weights = [p.weight for p in particles]
        total_weight = logsumexp(np.array(weights))
        weights_normalized = weights - total_weight

        # Compute log average weight (used if resampling, else only for record)
        avg_weight = total_weight - np.log(n_particles)
        if verbosity > 0:
            for i, p in enumerate(particles):
                print(
                    f'├ Particle {i:3d} (weight {p.weight:.4f}). `{p.context[-1]}` : {p}'
                )
            print(f'│ Step {step_num:3d} average weight: {avg_weight:.4f}')

        if return_record:
            step['particles'] = [
                {'context': p.context.copy(), 'weight': p.weight} for p in particles
            ]
            step['average_weight'] = avg_weight

        # Resample if necessary
        if -logsumexp(weights_normalized * 2) < np.log(ess_threshold) + np.log(
            n_particles
        ):
            # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
            probs = np.exp(weights_normalized)

            if return_record or verbosity > 0:
                # resampling: sample indices to copy
                resampled_indices = [
                    np.random.choice(range(len(particles)), p=probs)
                    for _ in range(n_particles)
                ]
                # resampled_indices.sort()  # removed. sorting should be done in post if necessary
                model.particles[request_id] = [
                    copy.deepcopy(particles[i]) for i in resampled_indices
                ]
                particles = model.particles[request_id]
                step['resample_indices'] = resampled_indices
            else:
                model.particles[request_id] = [
                    copy.deepcopy(
                        particles[np.random.choice(range(len(particles)), p=probs)]
                    )
                    for _ in range(n_particles)
                ]
                particles = model.particles[request_id]

            for p in particles:
                p.weight = avg_weight

            if verbosity > 0:
                print(
                    f'└╼  Resampling! {resampled_indices}. Weights all set to = {avg_weight:.4f}.'
                )
        else:
            if verbosity > 0:
                print('└╼')

        if return_record or verbosity > 0:
            step_num += 1
            if return_record:
                record['history'].append(step)

    return particles, record
