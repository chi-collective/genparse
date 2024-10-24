from itertools import product
import multiprocessing as mp
import os
from typing import Union

import pytest
import torch

from genparse.batch_inference.steer import ParticleApproximation
from genparse.util import InferenceSetup, set_seed


def _do_setup(
    model: str = 'mock-gpt2',
    proposal: str = 'character',
    use_parallel_proposal: bool = True,
    use_rust_parser: bool = True,
    seed: int = 0,
) -> InferenceSetup:
    # Use multiple processes only if parallel proposal is enabled
    num_processes: int
    if use_parallel_proposal:
        num_processes = (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else min(mp.cpu_count(), 2)
        )
        assert num_processes >= 2
    else:
        num_processes = 1

    proposal_opts = {}
    if proposal == 'token':
        proposal_opts['K'] = 5

    result = InferenceSetup(
        model_name=model,
        grammar='start: "Sequential Monte Carlo is " ( "good" | "bad" )',
        proposal_name=proposal,
        num_processes=num_processes,
        use_rust_parser=use_rust_parser,
        proposal_opts=proposal_opts,
        seed=seed,
    )

    return result


def _run_seeded_example(
    model: str = 'mock-gpt2',
    proposal: str = 'character',
    use_parallel_proposal: bool = True,
    inference_method: str = 'smc',
    use_rust_parser: bool = True,
    n_particles: int = 10,
) -> tuple[ParticleApproximation, ...]:
    assert n_particles > 0

    seed = 0
    model_ = _do_setup(
        model=model,
        proposal=proposal,
        use_parallel_proposal=use_parallel_proposal,
        use_rust_parser=use_rust_parser,
        seed=seed,
    )

    result = tuple(
        set_seed(seed)
        or model_(
            'Say something nice about SMC:',
            method=inference_method,
            n_particles=n_particles,
            verbosity=1,
        )
        for _ in range(2)
    )
    return result


def _seeded_example_id(
    model: str,
    proposal: str,
    inference_method: str,
    use_parallel_proposal: bool,
    use_rust_parser: bool,
) -> str:
    parts = [f'{model}', f'{proposal}', f'{inference_method}']
    parts.append('parallel' if use_parallel_proposal else 'sequential')
    parts.append('rustparser' if use_rust_parser else 'pyparser')
    result = '_'.join(parts)
    return result


def _is_fast_case(case) -> bool:
    model, proposal, inference_method, use_parallel_proposal, use_rust_parser = case
    return (
        model == 'mock-gpt2'
        and inference_method == 'smc'
        and proposal == 'character'
        and use_parallel_proposal
        and not use_rust_parser
    )


_testdata = list(
    product(
        ['mock-gpt2', 'gpt2'],
        ['character', 'token'],
        ['smc', 'is'],
        [True, False],
        [True, False],
    )
)
_testids = [_seeded_example_id(*args) for args in _testdata]
fast_cases = [case for case in _testdata if _is_fast_case(case)]
fast_ids = [id_ for id_, case in zip(_testids, _testdata) if _is_fast_case(case)]
slow_cases = [case for case in _testdata if not _is_fast_case(case)]
slow_ids = [id_ for id_, case in zip(_testids, _testdata) if not _is_fast_case(case)]


@pytest.mark.parametrize(
    'model,proposal,inference_method,use_parallel_proposal,use_rust_parser',
    fast_cases,
    ids=fast_ids,
)
def test_inference_is_consistent(
    model: str,
    proposal: str,
    inference_method: str,
    use_parallel_proposal: bool,
    use_rust_parser: bool,
) -> None:
    result1, result2 = _run_seeded_example(
        model=model,
        proposal=proposal,
        inference_method=inference_method,
        use_parallel_proposal=use_parallel_proposal,
        use_rust_parser=use_rust_parser,
    )
    assert result1.posterior == result2.posterior


@pytest.mark.skipif(os.getenv('RUN_SLOW_TESTS') != '1', reason='slow')
@pytest.mark.parametrize(
    'model,proposal,inference_method,use_parallel_proposal,use_rust_parser',
    slow_cases,
    ids=slow_ids,
)
def test_inference_is_consistent_slow(
    model: str,
    proposal: str,
    inference_method: str,
    use_parallel_proposal: bool,
    use_rust_parser: bool,
) -> None:
    result1, result2 = _run_seeded_example(
        model=model,
        proposal=proposal,
        inference_method=inference_method,
        use_parallel_proposal=use_parallel_proposal,
        use_rust_parser=use_rust_parser,
    )
    assert result1.posterior == result2.posterior
