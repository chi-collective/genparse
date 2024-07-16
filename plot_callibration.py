# %%
import numpy as np
import json
from tqdm import tqdm
import polars as pl

# %%
genparse_filename = 'llama-3-smc-standard-p10-b10-1000-schema.jsonl'
llm_filename = 'llama-3-sampling-n10-1000.jsonl'
# %%
import nltk; nltk.download('punkt')
from bench.spider.evaluator import Evaluator
from pathlib import Path

raw_spider_dir = Path('/home/leodu/genparse/bench/spider/data/spider')
evaluator = Evaluator(raw_spider_dir)


# %%
def partition(a, equiv):
    partitions = [] # Found partitions
    for e in a: # Loop over each element
        found = False # Note it is not yet part of a know partition
        for p in partitions:
            if equiv(e, p[0]): # Found a partition for it!
                p.append(e)
                found = True
                break
        if not found: # Make a new partition for it.
            partitions.append([e])
    return partitions


# %%

def compute_p(data_line, genparse=True):
    if genparse:
        particle_outputs = [''.join(particle['tokens'][:-1])
            for particle in data_line['particles']]
        weights = np.exp(np.array([particle['weight'] for particle in data_line['particles']]))
    else:
        particle_outputs = [particle['text']
            for particle in data_line['particles']]
        weights = np.exp(np.array([particle['cumulative_logprob'] for particle in data_line['particles']]))

    weights /= weights.sum()
    db_name = data_line['db_name']

    sql_outputs = [
        evaluator.get_eval(
            output,
            db_name
        )
        for output in particle_outputs
    ]

    outputs = [
        (p, s, w) for p, s, w in zip(particle_outputs, sql_outputs, weights)
        if s is not None
    ]
    Z = sum(w for _, _, w in outputs)
    outputs = [
        (p, s, w / Z) for p, s, w in outputs
    ]

    partitions = partition(
        outputs, 
        lambda x, y: x[1] == y[1])

    equiv_w = np.array([sum([p[2] for p in partition]) for partition in partitions])
    equiv_p = [p[0][0] for p in partitions]
    equiv_s = [p[0][1] for p in partitions]

    gt_eval = evaluator.get_eval(
        data_line['gold'],
        db_name
    )

    correct = [
        1 if gt_eval == s else 0
        for s in equiv_s
    ]

    return equiv_p, equiv_s, equiv_w, correct

# %%
def calibration_df(filename, genparse=True):
    with open(filename) as f:
        data = f.readlines()

    data_lines = [json.loads(d) for d in data]
    gt_probs = [
        compute_p(data_line, genparse)
        for data_line in tqdm(data_lines)
    ]

    probs = [
        w for gt_prob in gt_probs
        for w in gt_prob[2]
    ]
    accuracy = [
        c for gt_prob in gt_probs
        for c in gt_prob[3]
    ]

    return pl.DataFrame({
        'probability under model': probs,
        'accuracy': accuracy
    })

# %%
genparse_df = calibration_df(genparse_filename)

# %%
llm_df = calibration_df(llm_filename, genparse=False)

# %%
df = pl.concat((
    genparse_df.with_columns(
        model=pl.lit("GenParse, 10 particles")
    ),
    llm_df.with_columns(
        model=pl.lit("LLM, 10 particles")
    )
))


# %%
import seaborn as sns
import matplotlib.pyplot as plt
g = sns.lmplot(x="probability under model", y="accuracy", hue="model",
    data=df, logistic=True, truncate=False, scatter=False,
    legend_out=False)
ax = plt.gca()
ax.set_title(
        'Execution calibration on Spider')
plt.plot([0, 1], [0, 1], linewidth=2, linestyle='--', color='k' )
plt.savefig('spider_calibration.png', bbox_inches='tight', dpi=300)