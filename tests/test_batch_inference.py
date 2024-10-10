import os
import torch
import warnings
import numpy as np
import multiprocessing as mp
from functools import partial
from genparse.util import lark_guide, set_seed
from genparse.batch_inference.lm import use_default_sampler
from genparse.lm import VirtualTokenizedLLM, MockLLM, LazyProb
from genparse.batch_inference import (
    BatchLLM,
    BatchVLLM,
    ParallelCharacterProposal,
    ParallelTokenProposal,
    CharacterBatchProposal,
    TokenBatchProposal,
    BatchStepModel,
    smc,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def test_character_abc():
    set_seed(0)

    guide = lark_guide('start: "a" "b" "c"')
    sequential_llm = BatchLLM.from_name('gpt2')
    prompt = 'Generate "abc":'
    max_tokens = 10
    n_particles = 10
    want = {'abc▪': 1.0}

    # sequential proposal

    sequential_proposal = CharacterBatchProposal(llm=sequential_llm.llm, guide=guide)

    step_model = BatchStepModel(
        batch_proposal=sequential_proposal,
        batch_llm=sequential_llm,
        max_tokens=max_tokens,
        prompt=prompt,
    )

    have = smc(step_model, n_particles=n_particles, verbosity=1)
    have.posterior.assert_equal(want)

    # parallel proposal

    parallel_proposal = ParallelCharacterProposal(
        llm=sequential_llm.llm,
        guide=guide,
        num_processes=min(mp.cpu_count(), 2),
        max_n_particles=100,
        seed=0,
    )
    step_model = BatchStepModel(
        batch_proposal=parallel_proposal,
        batch_llm=sequential_llm,
        max_tokens=max_tokens,
        prompt=prompt,
    )

    have = smc(step_model, n_particles=n_particles, verbosity=1)
    have.posterior.assert_equal(want)

    step_model.cleanup()


def test_token_abc():
    set_seed(0)

    guide = lark_guide('start: "a" "b" "c"')
    sequential_llm = BatchLLM.from_name('gpt2')
    prompt = 'Generate "abc":'
    max_tokens = 10
    n_particles = 10
    want = {'abc▪': 1.0}

    # sequential proposal

    sequential_proposal = TokenBatchProposal(llm=sequential_llm.llm, guide=guide, K=5)
    step_model = BatchStepModel(
        batch_proposal=sequential_proposal,
        batch_llm=sequential_llm,
        max_tokens=max_tokens,
        prompt=prompt,
    )

    have = smc(step_model, n_particles=n_particles, verbosity=1)
    have.posterior.assert_equal(want)

    # parallel proposal

    parallel_proposal = ParallelTokenProposal(
        llm=sequential_llm.llm,
        guide=guide,
        K=5,
        num_processes=min(mp.cpu_count(), 2),
        max_n_particles=100,
        seed=0,
    )
    step_model = BatchStepModel(
        batch_proposal=parallel_proposal,
        batch_llm=sequential_llm,
        max_tokens=max_tokens,
        prompt=prompt,
    )

    have = smc(step_model, n_particles=n_particles, verbosity=1)
    have.posterior.assert_equal(want)

    step_model.cleanup()


def test_parallel_this_that():
    class ThisThatLM(MockLLM):
        def __init__(self):
            super().__init__(V=[' this', ' that', '▪'], eos='▪')

        def logp_next(self, context):
            assert isinstance(context, tuple)
            assert set(context) <= self.V, f'OOVs detected: {set(context) - self.V}'
            if context == ():
                p = np.array([0.5, 0.5, 0])
            elif context[-1] == ' this':
                p = np.array([0.5, 0, 0.5])
            elif context[-1] == ' that':
                p = np.array([0, 0.5, 0.5])
            else:
                raise ValueError(f'Unexpected context: {context}')

            return LazyProb(np.log(p), self._encode, self._decode)

    batch_llm = BatchLLM(ThisThatLM())

    guide = lark_guide('start: " this this this" | " that that that"')

    parallel_proposal = ParallelTokenProposal(
        llm=batch_llm.llm,
        guide=guide,
        K=None,
        num_processes=min(mp.cpu_count(), 2),
        max_n_particles=500,
        seed=0,
    )

    step_model = BatchStepModel(
        batch_proposal=parallel_proposal,
        batch_llm=batch_llm,
        max_tokens=10,
        prompt=(),
    )

    # test posterior multi-modality
    want = {' this this this▪': 0.5, ' that that that▪': 0.5}
    have = smc(step_model, n_particles=500, verbosity=1)
    have.posterior.assert_equal(want, tol=0.5)

    # test weight quality
    want = 0.5**4 + 0.5**4
    have = np.exp(have.log_ml)

    assert abs(have - want) < 1e-6, [have, want]

    step_model.cleanup()


##########################
# VLLM integration tests #
##########################


def test_vllm():
    from vllm import LLM

    set_seed(0)

    if not torch.cuda.is_available():
        warnings.warn('Skipping vllm inference tests because cuda is not available')
        return

    vllm_llm = LLM('gpt2')

    _test_vllm_inference_abc(vllm_llm)
    _test_vllm_scoring(vllm_llm)


def _test_vllm_inference_abc(vllm_llm):
    batch_llm = BatchVLLM(VirtualTokenizedLLM(vllm_llm.llm_engine))

    guide = lark_guide('start: "a" "b" "c"')
    prompt = 'Generate "abc":'
    max_tokens = 100
    n_particles = 10
    want = {'abc▪': 1.0}

    parallel_proposal = ParallelCharacterProposal(
        llm=batch_llm.llm,
        guide=guide,
        num_processes=min(mp.cpu_count(), 2),
        max_n_particles=100,
        seed=0,
    )
    step_model = BatchStepModel(
        batch_proposal=parallel_proposal,
        batch_llm=batch_llm,
        max_tokens=max_tokens,
        prompt=prompt,
    )

    have = smc(step_model, n_particles=n_particles, verbosity=1)
    have.posterior.assert_equal(want)

    step_model.cleanup()

    parallel_proposal = ParallelTokenProposal(
        llm=batch_llm.llm,
        guide=guide,
        K=5,
        num_processes=min(mp.cpu_count(), 2),
        max_n_particles=100,
        seed=0,
    )
    step_model = BatchStepModel(
        batch_proposal=parallel_proposal,
        batch_llm=batch_llm,
        max_tokens=max_tokens,
        prompt=prompt,
    )

    have = smc(step_model, n_particles=n_particles, verbosity=1)
    have.posterior.assert_equal(want)

    step_model.cleanup()


def reference_scorer(vllm_llm, prompts, token_ids, temperature):
    # Memory intensive implementation
    from vllm import SamplingParams

    tokenizer = vllm_llm.get_tokenizer()

    masks = []
    all_token_ids = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        all_token_ids.append(prompt_ids + token_ids)
        masks.append([0] * len(prompt_ids) + ([1] * len(token_ids)))

    inputs = tokenizer.batch_decode(all_token_ids)

    with use_default_sampler(vllm_llm):
        outputs = vllm_llm.generate(
            prompts=inputs,
            use_tqdm=False,
            sampling_params=SamplingParams(
                prompt_logprobs=0, max_tokens=1, temperature=temperature
            ),
        )

    logprobs = [
        sum(
            output.prompt_logprobs[j][token_id].logprob
            for j, token_id in enumerate(all_token_ids[i])
            if masks[i][j]
        )
        for i, output in enumerate(outputs)
    ]

    return logprobs


def _test_vllm_scoring(vllm_llm):
    rtol = 0.01  # abs(a - b) <= rtol * abs(b)
    batch_llm = BatchVLLM(VirtualTokenizedLLM(vllm_llm.llm_engine))

    tokenizer = batch_llm.llm.tokenizer
    reference = partial(reference_scorer, vllm_llm=vllm_llm)

    prompts = [
        'Repeat " this that them": this that them\nRepeat " this that them":',
        'Repeat " this them that": this that that\nRepeat " this that that":',
        'Repeat " this that": this that\nRepeat " this that":',
    ]
    token_ids = tokenizer.encode(' this that them', add_special_tokens=False)

    want = reference(prompts=prompts, token_ids=token_ids, temperature=1)
    have = batch_llm.batch_score_sequences(
        prompts=prompts, token_ids=token_ids, temperature=1
    )

    assert np.allclose(have, want, atol=0, rtol=rtol), [have, want]

    want = reference(prompts=prompts, token_ids=token_ids, temperature=1.75)
    have = batch_llm.batch_score_sequences(
        prompts=prompts, token_ids=token_ids, temperature=1.75
    )

    assert np.allclose(have, want, atol=0, rtol=rtol), [have, want]


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
