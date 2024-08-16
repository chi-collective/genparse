# %%
import pandas as pd

df = pd.read_parquet(
    'hf://datasets/BatsResearch/planetarium/data/train-00000-of-00001.parquet'
)
# %%
df.to_parquet('planetarium_train.parquet', index=False)
from benchmark.make_planetarium_prompt import make_prompt


# %%
import polars as pl

df = pl.DataFrame(df)
message = [
    {
        'role': 'user',
        'content': make_prompt(
            df.filter(pl.col('id') == 77237)['natural_language'], 'blocksworld'
        ),
    }
] + ([])

# %%
import os

HF_ACCESS_TOKEN = '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'
os.environ['HF_TOKEN'] = HF_ACCESS_TOKEN


# %%
import vllm
import transformers
import torch

torch.cuda.empty_cache()

model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
# llm = vllm.LLM(
#     model=model_name,
#     rope_scaling={'type': 'dynamic', 'factor': 8.0},
#     max_model_len=7760,
# )
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# %%
# prompt = tokenizer.decode(
#     tokenizer.apply_chat_template(message, add_generation_prompt=True)
# )

# %%
# n = 10
# sampling_params = vllm.SamplingParams(
#     n=10, temperature=1.0, max_tokens=1000, seed=0
# )

# %%
prompt = tokenizer.apply_chat_template(
    message, add_generation_prompt=False, tokenize=False
)
# %%
# llm_outputs = llm.generate(prompt, sampling_params)

# %%
import multiprocessing as mp
from bench.cache import ProposalCache
from genparse.experimental.batch_inference import (
    BatchVLLM,
    BatchStepModel,
    smc,
    importance_sampling,
)


def get_n_processes(particles, n_processes):
    """Determines the number of processes to use."""
    if n_processes is None:
        return min(particles, 10, mp.cpu_count() - 1)
    elif isinstance(n_processes, int):
        return n_processes
    else:
        raise ValueError(f'Invalid n_processes value: {n_processes}')


n_particles = 10

proposal_cache = ProposalCache(
    guide_cache_path='guide_cache.pkl', maxsize=1, max_mem_usage=0.7
)
print(f'Initialized {proposal_cache}')

n_processes = get_n_processes(n_particles, n_particles)

batch_llm = BatchVLLM.from_name(model_name)

# %%
parallel_proposal = proposal_cache.fetch_or_create_proposal(
    llm=batch_llm.llm,
    grammar=open('benchmark/grammars/blocksworld_whitespace.lark').read(),
    proposal_name='character',
    n_processes=n_processes,
)

step_model = BatchStepModel(
    batch_proposal=parallel_proposal,
    batch_llm=batch_llm,
    max_tokens=1000,
)

# %%
step_model.set_prompt(prompt)
# %%
method = smc
particles = smc(
    step_model, ess_threshold=0.9, n_particles=n_particles, return_record=True
)
# %%
import json

json.dump(particles.record, open('notes/smc_viz/Example.json', 'w'))
# %%
output = ''.join(particles[0].context[:-1])

# %%
import planetarium

planetarium.evaluate(
    df.filter(pl.col('id') == 77237).item(row=0, column='problem_pddl'), output
)
# %%

# %%
print(df.filter(pl.col('id') == 77237).item(row=0, column='problem_pddl'))
# %%
print(output)
# %%
particles
# %%
