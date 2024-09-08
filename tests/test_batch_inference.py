import torch
import warnings
import multiprocessing as mp
from genparse.lm import VirtualTokenizedLLM
from genparse.batch_inference.lm import LogitsGrouper
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
from genparse.util import lark_guide, set_seed
from functools import partial

import os

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
        num_processes=max(mp.cpu_count(), 2),
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
        num_processes=max(mp.cpu_count(), 2),
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
        num_processes=max(mp.cpu_count(), 2),
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
        num_processes=max(mp.cpu_count(), 2),
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
    from vllm.model_executor.layers.sampler import Sampler

    tokenizer = vllm_llm.get_tokenizer()

    masks = []
    all_token_ids = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        all_token_ids.append(prompt_ids + token_ids)
        masks.append([0] * len(prompt_ids) + ([1] * len(token_ids)))

    inputs = tokenizer.batch_decode(all_token_ids)

    vllm_llm.llm_engine.model_executor.driver_worker.model_runner.model.sampler = (
        Sampler()
    )

    outputs = vllm_llm.generate(
        prompts=inputs,
        use_tqdm=False,
        sampling_params=SamplingParams(
            prompt_logprobs=0, max_tokens=1, temperature=temperature
        ),
    )

    vllm_llm.llm_engine.model_executor.driver_worker.model_runner.model.sampler = (
        LogitsGrouper()
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
    tol = 0.5  # difference in *logprobs*
    batch_llm = BatchVLLM(VirtualTokenizedLLM(vllm_llm.llm_engine))

    tokenizer = batch_llm.llm.tokenizer
    reference = partial(reference_scorer, vllm_llm=vllm_llm)

    prompts = [
        'Repeat " this that them": this that them\nRepeat " this that them":',
        'Repeat " this them that": this them that\nRepeat " this them that":',
        'Repeat " that this": that this\nRepeat " that this":',
    ]
    sequences = [' this that them', ' this them that', ' that this']
    token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in sequences]

    for temperature in [0.25, 1, 1.75]:
        print()
        for i, _token_ids in enumerate(token_ids):
            wants = reference(
                prompts=prompts, token_ids=_token_ids, temperature=temperature
            )
            haves = batch_llm.batch_score_sequences(
                prompts=prompts, token_ids=_token_ids, temperature=temperature
            )
            for j, (want, have) in enumerate(zip(wants, haves)):
                print(repr(prompts[j]), repr(sequences[i]), want, have)
                assert abs(have - want) < tol, [have, want, temperature, i, j]


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
