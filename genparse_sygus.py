# %%
import pandas as pd

splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'valid': 'data/valid-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet',
}
df = pd.read_parquet(
    'hf://datasets/MilaWang/gad-bv4-no-grammar-3shots/' + splits['train']
)

# %%
df
# %%
from genparse.experimental.batch_inference import BatchVLLM
from genparse.lm import VirtualTokenizedLLM
from bench.run_inference import run_lm_inference, run_smc_inference
import transformers
import vllm
import json

seed = 1234

model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
n_particles = 10
prompts = df['query'].tolist()
max_tokens = 60
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


# %%
def sygus_prompt_to_chat(prompt):
    prompt_pieces = prompt.split('Question:\n')

    messages = [{'role': 'system', 'content': prompt_pieces[0]}]

    for prompt_piece in prompt_pieces[1:-1]:
        q, a = prompt_piece.split('Solution:\n')
        messages += [
            {'role': 'user', 'content': q},
            {'role': 'assistant', 'content': a},
        ]

    messages += [{'role': 'user', 'content': prompt_piece[-1]}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# %%
prompts = [sygus_prompt_to_chat(prompt) for prompt in prompts]

# %%
import os

HF_ACCESS_TOKEN = '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'
os.environ['HF_TOKEN'] = HF_ACCESS_TOKEN

llm = vllm.LLM(
    model=model_name,
    rope_scaling={'type': 'dynamic', 'factor': 8.0},
    max_model_len=7760,
)

# %%
grammar = open('benchmark/grammars/sygus.lark').read()
batch_llm = BatchVLLM(VirtualTokenizedLLM(llm.llm_engine))
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


# %%
lm_outputs = run_lm_inference(
    prompts, llm, n_particles=n_particles, seed=seed, max_tokens=max_tokens
)
results = {'lm': lm_outputs, 'smc': smc_outputs, 'is': is_outputs}

json.dump(results, open('sygus_results.json', 'w'))
json.dump(df.to_dicts(), open('sygus_test.json', 'w'))
