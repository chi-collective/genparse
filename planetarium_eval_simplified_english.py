# %%
import json
from lark.exceptions import UnexpectedInput, VisitError
import planetarium
import scipy
from tqdm import tqdm
from joblib import Parallel, delayed


results = json.load(open('planetarium_results_simplified_english.json'))
data = json.load(open('planetarium_test_simplified_english.json'))

# %%
data

# %%
data[0]

# %%
results['smc']

# %%
# results['smc'][0]['generations']
results['smc'][0]['weights']

# %%

results['lm'][0]['generations']

# %%

json.dump(
    results['smc'][0]['record'],
    open('notes/smc_viz/planetarium_simplified_english.json', 'w'),
)


# %%
smc_result = 'You have 5 blocks.\nYour arm is empty.\nb1 is clear.\nb2 is clear.\nb2 is on b3.\nb3 is on b4.\nb4 is on b5.\nb1 is on the table.\nb5 is on the table.'
gt_result = 'You have 5 blocks.\nYour arm is empty.\nb1 is clear.\nb1 is on the table.\nb2 is clear.\nb2 is on b3.\nb3 is on b4.\nb4 is on b5.\nb5 is on the table.\n'

sorted(smc_result.strip().split('\n')) == sorted(gt_result.strip().split('\n'))

# %%
results['smc'][0]['weights']


# %%
import numpy as np


def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))


def process_method(method):
    outputs = []
    if method == 'lpoe':
        method_results = results['is']
    else:
        method_results = results[method]

    for method_result, datum in tqdm(zip(method_results, data)):
        ground_truth = set(datum['simplified_english_output'].strip().split('\n'))
        generations = [set(g.strip().split('\n')) for g in method_result['generations']]

        correct = [
            jaccard_similarity(generation, ground_truth) for generation in generations
        ]

        if method == 'lpoe':
            N = len(correct)
            weights = method_result['weights']
            normalized_weights = [1 / N for _ in range(N)]
        else:
            weights = method_result['weights']
            weights = np.exp(weights)
            normalized_weights = scipy.special.softmax(method_result['weights'])
        posterior_weighted = sum([p * o for p, o in zip(normalized_weights, correct)])
        outputs.append(
            {
                'accuracy': posterior_weighted,
                'method': method,
                'id': datum['id'],
                'log_mean_weight': np.log(np.mean(weights)),
                'mean_log_weight': np.mean(np.log(weights)),
                'num_objects': datum['num_objects'],
                'goal_is_abstract': datum['goal_is_abstract'],
                'goal_num_propositions': datum['goal_num_propositions'],
                'init_num_propositions': datum['init_num_propositions'],
                'init_is_abstract': datum['init_is_abstract'],
            }
        )
    return outputs


# results = Parallel(n_jobs=5)(delayed(process_method)(i) for i in ['is', 'lpoe', 'smc', 'lm'])
result_outputs = [process_method(i) for i in ['is', 'lpoe', 'smc', 'lm']]

# %%
outputs = result_outputs[0] + result_outputs[1] + result_outputs[2] + result_outputs[3]

# %%
import polars as pl
from plotnine import ggplot, geom_boxplot, aes

df = pl.DataFrame(outputs)


# %%


# %%
df.sort('accuracy')

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
outputs
# %%
import polars as pl

df = pl.from_dicts(outputs)
df
# %%
df.filter(pl.col('method').is_in(['smc', 'is']))
# %%

smc_df = df.filter(pl.col('method') == 'smc')
lpoe_df = df.filter(pl.col('method') == 'lpoe')

# %%
# Compute the average accuracy for each id under smc_df
smc_avg_accuracy = smc_df.group_by('id').agg(
    pl.col('accuracy').mean().alias('avg_accuracy')
)
# Left join df onto smc_avg_accuracy by id
joined_df = smc_avg_accuracy.join(df, on='id', how='left')

# Compute accuracy - avg_accuracy
joined_df = joined_df.with_columns(
    (pl.col('accuracy') - pl.col('avg_accuracy')).alias('accuracy_difference')
)

print('Joined dataframe with accuracy difference:')
print(joined_df)

# Optional: Sort by accuracy difference in descending order
joined_df_sorted = joined_df.sort('accuracy_difference', descending=True)

print('\nSorted by accuracy difference (descending):')
print(joined_df_sorted)

# %%
# Calculate average accuracy_difference per method and id
avg_accuracy_diff = joined_df_sorted.group_by(['method', 'id']).agg(
    pl.col('accuracy_difference').mean().alias('avg_accuracy_difference')
)

print('Average accuracy difference per method and id:')
print(avg_accuracy_diff)

# Optional: Calculate overall average per method
method_avg_accuracy_diff = avg_accuracy_diff.group_by('method').agg(
    pl.col('avg_accuracy_difference').mean().alias('method_avg_accuracy_difference')
)

print('\nOverall average accuracy difference per method:')
print(method_avg_accuracy_diff)


# %%
from plotnine import ggplot, aes, scale_x_discrete, geom_boxplot, theme_minimal, labs

plot = (
    ggplot(avg_accuracy_diff)
    + geom_boxplot(aes(x='method', y='avg_accuracy_difference'))
    + scale_x_discrete(limits=['smc', 'is', 'lpoe', 'lm'])
    # + theme_minima()
    + labs(
        title='Average Accuracy Difference by Method',
        x='Method',
        y='Average Accuracy Difference',
    )
)

# Display the plot
print(plot)

# Save the plot
plot.save('avg_accuracy_difference_by_method.png', dpi=300, width=10, height=6)

# %%
smc_grouped = smc_df.group_by('id').agg(
    [
        pl.col('accuracy').mean().alias('mean_accuracy'),
        pl.col('accuracy').var().alias('variance_accuracy'),
        pl.col('log_mean_weight').mean().alias('mean_log_mean_weight'),
        pl.col('log_mean_weight').var().alias('variance_log_mean_weight'),
    ]
)

lpoe_grouped = lpoe_df.group_by('id').agg(
    [
        pl.col('accuracy').mean().alias('mean_accuracy'),
        pl.col('accuracy').var().alias('variance_accuracy'),
        pl.col('mean_log_weight').mean().alias('mean_log_mean_weight'),
        pl.col('mean_log_weight').var().alias('variance_log_mean_weight'),
    ]
)

print(smc_grouped)
print(lpoe_grouped)
# %%
# Join is_grouped and smc_grouped
joined_df = lpoe_grouped.join(smc_grouped, on='id', suffix='_smc')

# Calculate the difference of mean_log_mean_weight and sum of variances
result_df = joined_df.select(
    [
        pl.col('id'),
        (pl.col('mean_log_mean_weight') - pl.col('mean_log_mean_weight_smc')).alias(
            'diff_mean_log_mean_weight'
        ),
        (
            pl.col('variance_log_mean_weight') + pl.col('variance_log_mean_weight_smc')
        ).alias('sum_variances'),
        (pl.col('mean_accuracy') - pl.col('mean_accuracy_smc')).alias(
            'diff_mean_accuracy'
        ),
        (pl.col('variance_accuracy') + pl.col('variance_accuracy_smc')).alias(
            'sum_variances_accuracy'
        ),
    ]
)

print(result_df)

# Calculate and print the average difference and average sum of variances
avg_diff = result_df['diff_mean_log_mean_weight'].mean()
avg_sum_var = result_df['sum_variances'].mean()

print(f'Average difference in mean log mean weight: {avg_diff}')
print(f'Average sum of variances: {avg_sum_var}')

# %%
from plotnine import ggplot, aes, geom_point, geom_errorbar, geom_errorbarh, labs
from plotnine import theme_light, theme_538


result_df = result_df.with_columns(
    weight_error=np.sqrt(pl.col('sum_variances')),
    accuracy_error=np.sqrt(pl.col('sum_variances_accuracy')),
)
plot = (
    ggplot(result_df, aes(x='diff_mean_log_mean_weight', y='diff_mean_accuracy'))
    + geom_point()
    + geom_errorbarh(
        aes(
            xmin='diff_mean_log_mean_weight - weight_error',
            xmax='diff_mean_log_mean_weight + weight_error',
        ),
        height=0,
    )
    + geom_errorbar(
        aes(
            ymin='diff_mean_accuracy - accuracy_error',
            ymax='diff_mean_accuracy + accuracy_error',
        ),
        width=0,
    )
    + theme_minimal()
    + labs(
        title='Difference in Mean Log Weight vs. Difference in Accuracy',
        x='Difference in Log Weight',
        y='Difference in Accuracy',
    )
    + theme_538()
)
# Display the plot
print(plot)
# Optionally, save the plot
plot.save('diff_mean_log_mean_weight_vs_diff_mean_accuracy.png', dpi=300)


# %%
# Order result_df by diff_mean_log_mean_weight ascending
ordered_result_df = result_df.sort('diff_mean_log_mean_weight')

print('Ordered result_df:')
print(ordered_result_df)

# Display the top 10 rows
print('\nTop 10 rows with smallest difference:')
print(ordered_result_df.head(10))

# Display the bottom 10 rows
print('\nBottom 10 rows with largest difference:')
print(ordered_result_df.tail(10))

# %%
# Add a column with range 1 to length of dataframe
ordered_result_df = ordered_result_df.with_columns(
    pl.Series(name='rank', values=range(1, len(ordered_result_df) + 1))
)

print('Ordered result_df with rank column:')
print(ordered_result_df)


# %%
# Add a column for the error bars
ordered_result_df = ordered_result_df.with_columns(error=np.sqrt(pl.col('sum_variances')))

# Create the scatterplot with error bars
plot = (
    ggplot(ordered_result_df, aes(x='rank', y='diff_mean_log_mean_weight'))
    + geom_point()
    + geom_errorbar(
        aes(
            ymin='diff_mean_log_mean_weight - error',
            ymax='diff_mean_log_mean_weight + error',
        ),
        width=0.2,
    )
    + theme_minimal()
    + labs(title='Difference in Mean Log Weight', x='Rank', y='Difference (IS - SMC)')
)

# Display the plot
print(plot)
# Optionally, save the plot
plot.save('diff_mean_log_mean_weight_scatter.png', dpi=300)

# %%
ordered_result_df = result_df.sort('diff_mean_accuracy')
# %%
ordered_result_df
# %%
