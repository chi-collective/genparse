# %%
# %load_ext autoreload
# %autoreload 2
# %%
import os
import vllm
import polars as pl
import transformers
import json
from benchmark.make_planetarium_prompt import make_system_prompt
import bm25s

HF_ACCESS_TOKEN = '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'
os.environ['HF_TOKEN'] = HF_ACCESS_TOKEN


df = pl.read_parquet(
    'hf://datasets/BatsResearch/planetarium/data/train-00000-of-00001.parquet'
)

# %%
df = df.with_columns(
    pl.col('natural_language').str.split(by='Your goal').list.first().alias('init'),
)
# %%

seed = 1234
n_particles = 50
n_shots = 10
n_examples = 30
max_n_objects = 3
df = df.filter(
    (pl.col('domain') == 'blocksworld')
    & (pl.col('num_objects') <= max_n_objects)
    & (pl.col('goal_is_abstract') == 0)
    & (pl.col('init_is_abstract') == 0)
)

df = df.sample(fraction=1, shuffle=True)
test_df, train_df = df.head(n_examples), df.tail(-n_examples)
corpus = train_df['natural_language'].to_list()
retriever = bm25s.BM25()
retriever.index(bm25s.tokenize(corpus))


messages = [
    {
        'role': 'system',
        'content': make_system_prompt('blocksworld'),
    }
]


model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

prompts = []
for row in test_df.to_dicts():
    query = bm25s.tokenize(row['natural_language'])

    example_ids, scores = retriever.retrieve(query_tokens=query, k=n_shots)

    prompt_message = messages[:]
    for example_id in example_ids[0]:
        example_row = train_df.row(example_id, named=True)
        prompt_message += [
            {
                'role': 'user',
                'content': example_row['natural_language'],
            },
            {
                'role': 'assistant',
                'content': example_row['problem_pddl'],
            },
        ]

    prompt_message += [
        {
            'role': 'user',
            'content': row['natural_language'],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_message, tokenize=False, add_generation_prompt=True
    )
    prompt += row['problem_pddl'].split(' (:init')[0]
    prompts.append(prompt)


print(prompts[0])
json.dump(prompts, open('planetarium_prompts.json', 'w'))

# %%


from genparse.experimental.batch_inference import BatchVLLM
from genparse.lm import VirtualTokenizedLLM
from bench.run_inference import run_lm_inference, run_smc_inference

llm = vllm.LLM(
    model=model_name,
    rope_scaling={'type': 'dynamic', 'factor': 8.0},
    max_model_len=7760,
)

lm_outputs = run_lm_inference(prompts, llm, n_particles=n_particles, seed=seed)

grammar = open('benchmark/grammars/blocksworld_restricted.lark').read()
batch_llm = BatchVLLM(VirtualTokenizedLLM(llm.llm_engine))
smc_outputs = run_smc_inference(
    grammar, prompts, batch_llm, n_particles=n_particles, ess_threshold=0.5
)
is_outputs = run_smc_inference(
    grammar, prompts, batch_llm, n_particles=n_particles, ess_threshold=0.0
)

results = {'lm': lm_outputs, 'smc': smc_outputs, 'is': is_outputs}

json.dump(results, open('planetarium_results.json', 'w'))
json.dump(test_df.to_dicts(), open('planetarium_test.json', 'w'))
