"""
Language model steering methods
"""

import asyncio
import random
import warnings

import numpy as np
import torch
import transformers
from arsenal.maths import logsumexp, sample_dict

from hfppl import Model

from genparse import EOS
from genparse.inference import (
    TraceSWOR,
    importance_sampling,
    smc_standard,
    smc_standard_record,
    smc_steer,
)
from genparse.lm import LM
from genparse.semiring import Float
from genparse.util import format_table, normalize

# ____________________________________________________________________________________
#


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ____________________________________________________________________________________
#


class BruteForceGlobalProductOfExperts:
    def __init__(self, lm1, lm2, MAX_LENGTH):
        # Create a reference distribution for the global product of experts by
        # materializing the distrbution over strings up to a maximum length
        self.lm1 = lm1
        self.lm2 = lm2
        self.p1 = (
            lm1.cfg.cnf.language(MAX_LENGTH)
            .filter(lambda x: len(x) <= MAX_LENGTH)
            .normalize()
        )
        self.p2 = (
            lm2.cfg.cnf.language(MAX_LENGTH)
            .filter(lambda x: len(x) <= MAX_LENGTH)
            .normalize()
        )
        self.target = (self.p1 * self.p2).normalize()


# TODO: support early termination options
class generation_tree:
    def __init__(self, lm, **opts):
        tracer = TraceSWOR()
        D = Float.chart()
        while tracer.root.mass > 0:
            with tracer:
                s, p = lm.sample(draw=tracer, **opts)
                D[s] += p
        D = Float.chart((k, D[k]) for k in sorted(D))
        self.D = D
        self.tracer = tracer

    def _repr_html_(self):
        return format_table([[self.D, self.tracer]])


# ____________________________________________________________________________________
#


class LocalProduct(LM):
    """This class implements a *local* product of experts, an LM that is derived by
    multiplying the conditional distributions of each token in a pair of
    token-synchronized LM.

    Typically, `LocalProduct` is a baseline method or a proposal distribution
    for the *global* product of experts.

    [Some people call LocalProduct the "locally optimal proposal distribution" -
    what does it actually optimize?]

    """

    def __init__(self, lm1, lm2):
        self.lm1 = lm1
        self.lm2 = lm2
        assert lm1.V == lm2.V
        assert lm1.eos == lm2.eos
        super().__init__(V=lm1.V, eos=lm1.eos)

    def __call__(self, ys):
        assert ys[-1] == self.eos
        p = 1
        for t in range(len(ys)):
            p *= self.p_next(ys[:t])[ys[t]]
        return p

    def p_next(self, ys):
        p1 = self.lm1.p_next(ys)
        p2 = self.lm2.p_next(ys)

        # TODO: p_next should already be normalized!  Skipping the normalization
        # below would allow energy-based models.
        p1 = normalize(p1)
        p2 = normalize(p2)

        # Below, we could alternatively use p2's support; any `k` that's not in
        # both must have probability zero.
        return (p1 * p2).normalize()


# _______________________________________________________________________________
# Approximate inference


def run(lm1, lm2, *, MAX_LENGTH, n_particles, METHOD):
    # TODO: I'd like to have a target--proposal pair passed in and for the SMC
    # stuff to combine it in the right way.  If we pass an unnormalized target
    # (i.e., an energy), then we get a consistent semantics (i.e., we are just
    # off by the normalization constant everywhere).

    # This interface is used in HFPPL / LLamPPL
    class Particle:
        def __init__(self, ys=None):
            self.ys = ys
            self.weight = 0.0

            self.Q = 0.0

        def start(self):
            self.ys = []

        def done_stepping(self):
            return EOS in self.ys

        def untwist(self):  # unused
            pass

        async def step(self):
            ys = tuple(self.ys)

            p1 = lm1.p_next(ys)
            p2 = lm2.p_next(ys)

            # TODO: p_next should already be normalized!  Skipping the
            # normalization below would allow energy-based models.
            p1 = p1.normalize()
            p2 = p2.normalize()

            # assert np.allclose(p1.sum(), 1), p1.sum()
            # assert np.allclose(p2.sum(), 1), p2.sum()

            q_ = p1 * p2

            Z = q_.sum()

            q = q_.normalize()

            if len(ys) > MAX_LENGTH:
                warnings.warn('force </s>')
                y = EOS

            else:
                y = sample_dict(q)

            # self.weight += np.log(p1[y] * p2[y] / (q[y] / Z))
            # self.weight += np.log(p1[y]) + np.log(p2[y]) - np.log(q[y]) + np.log(Z)
            # self.weight += np.log(Z)

            self.weight += np.log(Z)
            self.Q += np.log(q[y]) if q[y] > 0 else -np.inf

            # self.weight += np.log(p1(token | history) / p2(prev_token | prev_history))

            self.ys.append(y)

        def __repr__(self):
            return repr(self.ys)

    if METHOD == 'is':
        return asyncio.run(importance_sampling(Particle(), n_particles=n_particles))
    elif METHOD == 'smc-steer':
        return asyncio.run(smc_steer(Particle(), n_particles=n_particles, n_beam=1))
    elif METHOD == 'smc-standard':
        return asyncio.run(smc_standard(Particle(), n_particles=n_particles))
    else:
        raise AssertionError(METHOD)


class VLLMParticle(Model):
    def __init__(self, llm, guide, proposal, prompt, max_tokens, verbosity=0):
        from vllm.sampling_params import SamplingParams
        from genparse.tokenization import decode_tokenizer_vocab


        super().__init__()
        self.llm = llm # type: async
        # self.llm._model: VLLM
        # call the vllm engine of VLLM
        inputs = self.llm._model._convert_v1_inputs(prompts=prompt, prompt_token_ids=None)
        # add request for prompt
        self.llm._model._validate_and_add_requests(
            inputs=inputs,
            # using default params because we only rely on logits
            params=SamplingParams(max_tokens=max_tokens, stop_token_ids=[self.llm.eos]),
            lora_request=None
        )
        self.token_to_id = {x: i for i, x in enumerate(decode_tokenizer_vocab(self.llm.tokenizer))}

        self.guide = guide
        self.prompt = prompt
        self.context = []
        
        self.sampler_output = []

        self.proposal = proposal
        self.max_tokens = max_tokens
        
        
        self.verbosity = verbosity


    async def step(self):
        from vllm.sequence import (
            CompletionSequenceGroupOutput, SequenceOutput, 
            ExecuteModelRequest, SamplerOutput, Logprob
        )

        seq_group_metadata_list, scheduler_outputs = self.llm._model.llm_engine.scheduler.schedule()
        
        if scheduler_outputs.is_empty():
            # finishing due to resource
            self.llm._model.llm_engine._process_model_outputs(
                [], scheduler_outputs.scheduled_seq_groups,
                scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)
            self.llm._model.llm_engine.model_executor.stop_remote_worker_execution_loop()
            self.finish()

            return

        execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
            num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
            running_queue_size=scheduler_outputs.running_queue_size,
        )

        # executing:
        (token, weight_update) = await self.proposal.sample_next_token(
            prompt=self.prompt,
            context=''.join(self.context),
            compare_time=(self.verbosity > 1),
            execute_model_req=execute_model_req
        )
        token_id = self.token_to_id.get(token, self.llm._model.eos_token_id)
        self.context.append(token)
        self.weight += np.log(weight_update)
        self.max_tokens -= 1
        print("token, token_id", token, token_id)
        if self.verbosity > 1:
            
            print(f"`{token}` : {''.join(self.context)} : {self.weight}")

        # TODO: fit token back into SampleOutput, so vllm can
        # process it correctly
        output = []
        for seq_group_md in seq_group_metadata_list:
            parent_seq_id = list(seq_group_md.seq_data.keys())[0]
            output.append(SamplerOutput(
                outputs=[
                    CompletionSequenceGroupOutput(
                        samples=[SequenceOutput(parent_seq_id=parent_seq_id, output_token=token_id, logprobs={token_id: Logprob(logprob=np.log(weight_update))})],
                        prompt_logprobs=None)]
            ))
        self.sampler_output = self.sampler_output + output

        # post-processing. Do after sample is chosen
        self.llm._model.llm_engine._process_model_outputs(
            output, scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.ignored_seq_groups, seq_group_metadata_list)

        # Log stats.
        self.llm._model.llm_engine.do_log_stats(scheduler_outputs, output)

        return


    def immutable_properties(self):
        return ['llm', 'prompt', 'guide', 'verbosity']

    def __repr__(self):
        return f"`{'' if not self.context else self.context[-1]}` : {''.join(self.context)} : {self.weight}"

    def __str__(self):
        return ''.join(self.context)


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

        model = VLLMParticle(
            llm=self.llm,
            guide=self.guide,
            prompt=prompt,
            proposal=proposal,
            max_tokens=max_tokens,
            verbosity=verbosity,
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
                particles = asyncio.run(smc_standard(model, n_particles=n_particles))

        elif method == 'importance-sampling':
            particles = asyncio.run(importance_sampling(model, n_particles=n_particles))

        else:
            raise ValueError(f'Unknown inference method: {method}.')

        return ParticleApproximation(particles), record



# ____________________________________________________________________________________
# Approximate inference with HFPPL
# This code is still experimental and actively being developed
# TODO: write tests

class HFPPLParticle(Model):
    """
    Simple HFPPL model (particle).
    TODO: Create a proposal interface to make this class reusable.
        I think this class should be essentially hidden to the user and only be used by the HFPPLSampler.
    """

    def __init__(self, llm, guide, proposal, prompt, max_tokens, verbosity=0):
        super().__init__()
        self.llm = llm
        self.guide = guide
        self.prompt = prompt
        self.context = []
        self.proposal = proposal
        self.max_tokens = max_tokens
        self.verbosity = verbosity

    async def step(self):
        (token, weight_update) = await self.proposal.sample_next_token(
            prompt=self.prompt,
            context=''.join(self.context),
            compare_time=(self.verbosity > 1),
        )
        self.context.append(token)
        self.weight += np.log(weight_update)
        self.max_tokens -= 1

        if self.verbosity > 1:
            print(f"`{token}` : {''.join(self.context)} : {self.weight}")

        if token == self.llm.eos or self.max_tokens == 0 or token == EOS:
            self.finish()
            return

    def immutable_properties(self):
        return ['llm', 'prompt', 'guide', 'verbosity']

    def __repr__(self):
        return f"`{'' if not self.context else self.context[-1]}` : {''.join(self.context)} : {self.weight}"

    def __str__(self):
        return ''.join(self.context)


class HFPPLSampler:
    def __init__(self, llm, guide):
        """
        Args:
            llm (AsyncGreedilyTokenizedLLM)
            guide (LM)
        Returns:
            particle_approximation (ParticleApproximation)
            record (dict | NoneType): information about the run
        """
        self.llm = llm
        self.guide = guide

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

        model = HFPPLParticle(
            llm=self.llm,
            guide=self.guide,
            prompt=prompt,
            proposal=proposal,
            max_tokens=max_tokens,
            verbosity=verbosity,
        )

        record = None
        if method == 'smc-steer':
            assert n_beam is not None
            if return_record:
                warnings.warn('Record not yet implemented for smc-steer')
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
                particles = asyncio.run(smc_standard(model, n_particles=n_particles))

        elif method == 'importance-sampling':
            particles = asyncio.run(importance_sampling(model, n_particles=n_particles))

        else:
            raise ValueError(f'Unknown inference method: {method}.')

        return ParticleApproximation(particles), record


class ParticleApproximation:
    def __init__(self, particles):
        self.particles = particles
        self.log_weights = [p.weight for p in self.particles]
        self.log_ml = logsumexp(self.log_weights) - np.log(len(self.log_weights))
        self._compute_posterior()

    def __iter__(self):
        return iter(self.particles)

    def _compute_posterior(self):
        self.posterior = Float.chart()
        for p in self.particles:
            self.posterior[str(p)] += np.exp(p.weight)
        self.posterior = self.posterior.normalize()

    def sample(self, n=None, draw=sample_dict):
        if n is None:
            return draw(self.posterior)
        else:
            return [draw(self.posterior) for _ in range(n)]
