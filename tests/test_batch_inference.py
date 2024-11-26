import gc
import torch
import warnings
import multiprocessing as mp
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


def test_character_abc():
    set_seed(0)

    guide = lark_guide('start: "a" "b" "c"')
    sequential_llm = BatchLLM.from_name('gpt2')
    prompt = 'Generate "abc":'
    max_tokens = 100
    n_particles = 10
    want = {f'abc{sequential_llm.llm.eos}': 1.0}

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
    del sequential_llm.llm._model.model
    gc.collect()
    torch.cuda.empty_cache()


def test_token_abc():
    set_seed(0)

    guide = lark_guide('start: "a" "b" "c"')
    sequential_llm = BatchLLM.from_name('gpt2')
    prompt = 'Generate "abc":'
    max_tokens = 100
    n_particles = 10
    want = {f'abc{sequential_llm.llm.eos}': 1.0}

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
    del sequential_llm.llm._model.model
    gc.collect()
    torch.cuda.empty_cache()


def test_vllm_abc():
    set_seed(0)

    if not torch.cuda.is_available():
        warnings.warn('Skipping vllm test because cuda is not available')
        return

    guide = lark_guide('start: "a" "b" "c"')
    batch_llm = BatchVLLM.from_name('gpt2')
    prompt = 'Generate "abc":'
    max_tokens = 100
    n_particles = 10
    want = {f'abc{batch_llm.llm.eos}': 1.0}

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

    have = smc(step_model, n_particles=n_particles)
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

    have = smc(step_model, n_particles=n_particles)
    have.posterior.assert_equal(want)

    step_model.cleanup()
    batch_llm.free_vllm_gpu_memory()


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
