import gc

import torch
import pytest

from genparse import InferenceSetup


def get_inference_setup():
    grammar = """
    start: "Sequential Monte Carlo is " ( "good" | "bad" )
    """
    return InferenceSetup('gpt2', grammar, proposal_name='character')


# Reproduce the free_vllm_memory logic here so that we can run this benchmark with GPU on old
# commits for benchmark prototyping purposes.
def cleanup(inference_setup):
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )

        destroy_model_parallel()
        destroy_distributed_environment()

        try:
            del inference_setup.llm.llm_engine.model_executor
        except AttributeError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        pass


def get_and_clean_up_inference_setup():
    setup = get_inference_setup()
    cleanup(setup)


def do_inference(inference_setup_):
    return inference_setup_(' ', n_particles=5, verbosity=1)


@pytest.mark.benchmark()
def test_tiny_example_setup(benchmark):
    benchmark(get_and_clean_up_inference_setup)


@pytest.mark.benchmark()
def test_tiny_example_inference(benchmark):
    inference_setup = get_inference_setup()
    benchmark(do_inference, inference_setup)
    cleanup(inference_setup)
