# %%
import json
from lark.exceptions import UnexpectedInput, VisitError
import planetarium
import scipy
from tqdm import tqdm
from joblib import Parallel, delayed


results = json.load(open('planetarium_results.json'))
data = json.load(open('planetarium_test.json'))


def get_correct(outputs, gt_output):
    results = []
    for output in outputs:
        try:
            results.append(planetarium.evaluate(output, gt_output)[2])

        except (UnexpectedInput, ValueError, VisitError, AttributeError):
            # except:
            # you can get unexpected input for ungrammatical inputs
            # ValueError for referencing an object not in the object list
            # VisitError for duplicate object names
            results.append(False)
    return results


# %%
def process_method(method):
    outputs = []
    if method == 'lpoe':
        method_results = results['is']
    else:
        method_results = results[method]

    for method_result, datum in tqdm(zip(method_results, data)):
        generations = [
            datum['problem_pddl'].split(' (:init')[0] + output
            for output in method_result['generations']
        ]
        correct = get_correct(generations, datum['problem_pddl'])

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
                'id': datum['id'],
                'num_objects': datum['num_objects'],
                'goal_is_abstract': datum['goal_is_abstract'],
                'goal_num_propositions': datum['goal_num_propositions'],
                'init_num_propositions': datum['init_num_propositions'],
                'init_is_abstract': datum['init_is_abstract'],
            }
        )
    return outputs


results = Parallel(n_jobs=5)(
    delayed(process_method)(i) for i in ['is', 'lpoe', 'smc', 'lm']
)
# results = [process_method(i) for i in ['is', 'lpoe', 'smc', 'lm']]

# %%
outputs = results[0] + results[1] + results[2] + results[3]

# %%
import pandas as pd
from plotnine import ggplot, geom_boxplot, aes

df = pd.DataFrame(outputs)

# %%
import pandas as pd
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
