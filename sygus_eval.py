# %%
import json
from lark.exceptions import UnexpectedInput, VisitError
import planetarium
import scipy
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd


results = json.load(open('sygus_results.json'))
splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'valid': 'data/valid-00000-of-00001.parquet',
    'test': 'data/test-00000-of-00001.parquet',
}
data = pd.read_parquet(
    'hf://datasets/MilaWang/gad-bv4-no-grammar-3shots/' + splits['train']
)


# %%
def process_method(method):
    outputs = []
    if method == 'lpoe':
        method_results = results['is']
    else:
        method_results = results[method]

    for method_result, datum in tqdm(zip(method_results, data['answer'])):
        correct = [generation == datum for generation in method_result['generations']]

        if method == 'lpoe':
            N = len(correct)
            weights = [1 / N for _ in range(N)]
        else:
            weights = method_result['weights']
            weights = scipy.special.softmax(method_result['weights'])
        posterior_weighted = sum([p * o for p, o in zip(weights, correct)])
        outputs.append(
            {
                'accuracy': posterior_weighted,
                'method': method,
            }
        )
    return outputs


result_outputs = [process_method(i) for i in ['is', 'lpoe', 'smc']]

# %%
outputs = result_outputs[0] + result_outputs[1] + result_outputs[2]  # + results[3]

# %%
from plotnine import ggplot, geom_boxplot, aes

df = pd.DataFrame(outputs)

# %%
from plotnine import geom_bar


# Gallery, distributions
(ggplot(df) + geom_boxplot(aes(x='method', y='accuracy')))
# %%
df[df['method'] == 'is']['accuracy'].mean()
# %%
df[df['method'] == 'lpoe']['accuracy'].mean()
# %%
df[df['method'] == 'smc']['accuracy'].mean()
# %%
df[df['method'] == 'lm']['accuracy'].mean()
# %%
df[df['method'] == 'is']['accuracy'].std()
# %%
df[df['method'] == 'lpoe']['accuracy'].std()
# %%
df[df['method'] == 'smc']['accuracy'].std()
# %%
df[df['method'] == 'lm']['accuracy'].std()
# %%
df
# %%
df[df['method'] == 'smc']
# %%
df[df['method'] == 'lpoe']


# %%
results['smc']
# %%
test_data = pd.read_parquet(
    'hf://datasets/MilaWang/gad-bv4-no-grammar-3shots/' + splits['test']
)
test_data
# %%
valid_data = pd.read_parquet(
    'hf://datasets/MilaWang/gad-bv4-no-grammar-3shots/' + splits['valid']
)
valid_data
# %%
