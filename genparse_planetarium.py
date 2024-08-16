# %%
# %load_ext autoreload
# %autoreload 2
# %%
import os
import pandas as pd
import vllm
import polars as pl
import transformers
from benchmark.make_planetarium_prompt import make_system_prompt

HF_ACCESS_TOKEN = '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'
os.environ['HF_TOKEN'] = HF_ACCESS_TOKEN


df = pd.read_parquet(
    'hf://datasets/BatsResearch/planetarium/data/train-00000-of-00001.parquet'
)
# %%
df.to_parquet('planetarium_train.parquet', index=False)
df = pl.DataFrame(df)
# %%

seed = 0
n_particles = 10
n_shots = 5
n_examples = 5
max_n_objects = 5
rows = df.filter(
    (pl.col('domain') == 'blocksworld') & (pl.col('num_objects') <= max_n_objects)
).sample(n=n_shots + n_examples, seed=seed)

messages = [
    {
        'role': 'system',
        'content': make_system_prompt('blocksworld'),
    }
]

for row in rows.to_dicts()[:n_shots]:
    messages += [
        {
            'role': 'user',
            'content': row['natural_language'],
        },
        {
            'role': 'assistant',
            'content': row['problem_pddl'],
        },
    ]


model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

prompts = []
for row in rows.to_dicts()[n_shots:]:
    prompt_message = messages + [
        {
            'role': 'user',
            'content': row['natural_language'],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False
    )
    prompts.append(prompt)


from genparse.experimental.batch_inference import BatchVLLM
from genparse.lm import VirtualTokenizedLLM
from bench.run_inference import run_lm_inference, run_smc_inference

llm = vllm.LLM(
    model=model_name,
    rope_scaling={'type': 'dynamic', 'factor': 8.0},
    max_model_len=7760,
)

lm_outputs = run_lm_inference(prompts, llm, n_particles=n_particles, seed=seed)

grammar = open('benchmark/grammars/blocksworld_whitespace.lark').read()
batch_llm = BatchVLLM(VirtualTokenizedLLM(llm.llm_engine))
smc_outputs = run_smc_inference(
    grammar, prompts, batch_llm, n_particles=10, ess_threshold=0.9
)
is_outputs = run_smc_inference(
    grammar, prompts, batch_llm, n_particles=10, ess_threshold=0.0
)

import ipdb

ipdb.set_trace()

results = {'lm': lm_outputs, 'smc': smc_outputs, 'is': is_outputs}

import json

json.dump(results, open('planetarium_results.json', 'w'))
json.dump(rows.to_dicts()[n_shots:], open('planetarium_test.json', 'w'))
