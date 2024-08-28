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
n_shots = 0
n_replicates = 10
max_n_objects = 5
max_tokens = 110
explicit_df = df.filter(
    (pl.col('domain') == 'blocksworld')
    & (pl.col('num_objects') <= max_n_objects)
    & (pl.col('goal_is_abstract') == 0)
    & (pl.col('init_is_abstract') == 0)
)
abstract_df = df.filter(
    (pl.col('domain') == 'blocksworld')
    & (pl.col('num_objects') <= max_n_objects)
    & (pl.col('goal_is_abstract') == 1)
    & (pl.col('init_is_abstract') == 1)
)

# %%
# Join explicit_df and abstract_df using problem_pddl as the key
# simplified_english_df = explicit_df.join(
#     abstract_df,
#     on='problem_pddl',
#     how='inner',
#     suffix='_abstract'
# )
simplified_english_df = explicit_df

# %%
print(len(simplified_english_df))
simplified_english_df = simplified_english_df.unique(subset=['init'])
print(len(simplified_english_df))
n_examples = len(simplified_english_df)
# n_examples = 1


# %%
df = simplified_english_df.rename({'init': 'simplified_english_output'})

# %%
df = df.sample(fraction=1, shuffle=True)
test_df, train_df = df.head(n_examples), df.tail(-n_examples)
corpus = train_df['problem_pddl'].to_list()
retriever = bm25s.BM25()
retriever.index(bm25s.tokenize(corpus))


messages = [
    {
        'role': 'system',
        'content': "You write natural language descriptions of blocksworld planning problems. Given a PDDL description, your objective is to write a simplified natural language description of each of the propositions for the domain's initial conditions (the propositions inside the ':init' parenthetical).",
    }
]


model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

prompts = []
for row in test_df.to_dicts():
    query = bm25s.tokenize(row['problem_pddl'])

    example_ids, scores = retriever.retrieve(query_tokens=query, k=n_shots)

    prompt_message = messages[:]
    for example_id in example_ids[0]:
        example_row = train_df.row(example_id, named=True)
        prompt_message += [
            {
                'role': 'user',
                'content': example_row['problem_pddl'],
            },
            {
                'role': 'assistant',
                'content': example_row['natural_language'],
                # 'content': example_row['simplified_english_output'],
            },
        ]

    prompt_message += [
        {
            'role': 'user',
            'content': row['problem_pddl'],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        prompt_message, tokenize=False, add_generation_prompt=True
    )
    prompts.append(prompt)


print(prompts[0])
json.dump(prompts, open('planetarium_simplified_english_prompts.json', 'w'))

prompts = prompts * n_replicates


# %%
from genparse.experimental.batch_inference import BatchVLLM
from genparse.lm import VirtualTokenizedLLM
from bench.run_inference import run_lm_inference, run_smc_inference

llm = vllm.LLM(
    model=model_name,
    rope_scaling={'type': 'dynamic', 'factor': 8.0},
    max_model_len=7760,
)

# %%
lm_outputs = run_lm_inference(prompts, llm, n_particles=n_particles, seed=seed)

# %%
grammar = open('benchmark/grammars/blocksworld_init_simplified_english.lark').read()

# %%
batch_llm = BatchVLLM(VirtualTokenizedLLM(llm.llm_engine))

# %%
# from genparse.util import lark_guide
# guide = lark_guide(grammar)
# lm_generation = 'You have 3 blocks.\nYou are holding b1.\nb2 is clear.\nb2 is on the table.\nb3 is clear.'

# # %%
# guide.p_next(lm_generation)

# %%
smc_outputs = run_smc_inference(
    grammar,
    prompts,
    batch_llm,
    n_particles=n_particles,
    ess_threshold=0.5,
    max_tokens=max_tokens,
)
is_outputs = run_smc_inference(
    grammar,
    prompts,
    batch_llm,
    n_particles=n_particles,
    ess_threshold=0.0,
    max_tokens=max_tokens,
)

results = {'lm': lm_outputs, 'smc': smc_outputs, 'is': is_outputs}

json.dump(results, open('planetarium_results_simplified_english.json', 'w'))

data = test_df.to_dicts()
data = data * n_replicates
json.dump(data, open('planetarium_test_simplified_english.json', 'w'))
# %%

# encodings = [tokenizer.encode(x) for x in df["simplified_english_output"]]

# # %%
# encoding_lens = [len(x) for x in encodings]
# max(encoding_lens)

# # %%
# df
# # %%
# max(df["init_num_propositions"])
# # %%
# max(df["goal_num_propositions"])

# # %%
# lm_outputs
# # %%
# print(prompts[0])
# # %%
