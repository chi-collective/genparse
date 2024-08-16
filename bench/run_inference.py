from bench.cache import ProposalCache
from genparse.experimental.batch_inference import (
    BatchStepModel,
    smc,
)
import multiprocessing as mp
import vllm


def get_n_processes(particles, n_processes):
    """Determines the number of processes to use."""
    if n_processes is None:
        return min(particles, 10, mp.cpu_count() - 1)
    elif isinstance(n_processes, int):
        return n_processes
    else:
        raise ValueError(f'Invalid n_processes value: {n_processes}')


def run_lm_inference(
    prompt_list,
    llm,
    method='sampling',
    n_particles=10,
    max_tokens=1000,
    best_of=3,
    seed=0,
):
    if method == 'sampling':
        sampling_params = vllm.SamplingParams(
            n=n_particles, temperature=1.0, max_tokens=max_tokens, seed=seed
        )
    elif method == 'greedy':
        sampling_params = vllm.SamplingParams(
            best_of=best_of, temperature=0.0, max_tokens=max_tokens, seed=seed
        )
    elif method == 'beam':
        sampling_params = vllm.SamplingParams(
            best_of=best_of, temperature=0.0, max_tokens=max_tokens, seed=seed
        )
    else:
        raise ValueError(f'Invalid method: {method}')

    llm_outputs = llm.generate(prompt_list, sampling_params)
    output = [
        {
            'generations': [o.text for o in example.outputs],
            'weights': [0 for _ in example.outputs],
            'cumulative_logprob': [o.cumulative_logprob for o in example.outputs],
        }
        for example in llm_outputs
    ]
    return output


def run_smc_inference(
    grammar,
    prompt_list,
    batch_llm,
    n_particles=10,
    ess_threshold=0.5,
    max_size=1,
    max_mem_usage=0.7,
    max_tokens=1000,
    proposal='character',
    proposal_args={},
):
    proposal_cache = ProposalCache(
        guide_cache_path='guide_cache.pkl', maxsize=max_size, max_mem_usage=max_mem_usage
    )

    n_processes = get_n_processes(n_particles, n_particles)

    # %%
    parallel_proposal = proposal_cache.fetch_or_create_proposal(
        llm=batch_llm.llm,
        grammar=grammar,
        proposal_name=proposal,
        proposal_args=proposal_args,
        n_processes=n_processes,
    )

    step_model = BatchStepModel(
        batch_proposal=parallel_proposal,
        batch_llm=batch_llm,
        max_tokens=max_tokens,
    )

    return [
        run_smc(step_model, prompt, n_particles, ess_threshold) for prompt in prompt_list
    ]


def run_smc(model, prompt, n_particles=10, ess_threshold=0.5):
    model.set_prompt(prompt)
    particles = smc(
        model, ess_threshold=ess_threshold, n_particles=n_particles, return_record=True
    )
    generations = [
        ''.join(p['context'][:-1]) for p in particles.record['history'][-1]['particles']
    ]
    weights = [p['weight'] for p in particles.record['history'][-1]['particles']]
    return {'generations': generations, 'weights': weights, 'record': particles.record}
