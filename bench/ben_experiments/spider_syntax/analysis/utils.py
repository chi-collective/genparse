import os
import json
import warnings


def binomial_ci(data, col):
    data = data[col]
    n = len(data)
    mean = np.mean(data)
    ci_low, ci_upp = binom.interval(0.95, n, mean)
    return {f'{col}_mean': mean, f'{col}_lwr': ci_low / n, f'{col}_upr': ci_upp / n}


def mean_cl(data, col):
    data = data[col]
    mean = np.mean(data)
    std_err = np.std(data) / np.sqrt(len(data))
    ci_low = mean - 1.96 * std_err
    ci_upp = mean + 1.96 * std_err
    return {f'{col}_mean': mean, f'{col}_lwr': ci_low, f'{col}_upr': ci_upp}


def melt_and_add_confidence_intervals(df, id_vars, value_vars):
    df_melted = df.melt(
        id_vars=id_vars, value_vars=value_vars, var_name='metric', value_name='mean'
    )

    def get_confidence_interval(row, conf_type):
        condition = pd.Series(True, index=df.index)
        for var in id_vars:
            condition &= df[var] == row[var]
        return df.loc[condition, row['metric'].replace('_mean', conf_type)].values[0]

    df_melted['lwr'] = df_melted.apply(
        lambda x: get_confidence_interval(x, '_lwr'), axis=1
    )
    df_melted['upr'] = df_melted.apply(
        lambda x: get_confidence_interval(x, '_upr'), axis=1
    )

    return df_melted


def count_resample_steps(record):
    steps = record['history']
    n = 0
    for step in steps:
        if 'resample_indices' in step:
            n += 1
    return n


def make_file_path(results_dir, model, method, ess, run, n_particles, proposal=None):
    ess = f'-{ess}' if ess is not None else ''
    if proposal is None:
        # baseline model
        raise NotImplementedError
    else:
        return os.path.join(
            results_dir,
            model,
            proposal,
            method,
            f'{model}-{method}-{run}{ess}-p{n_particles}-{proposal}.jsonl',
        )


import itertools


def _iter_args(models, methods, runs, ess_thresholds, n_particles_list, proposals):
    return itertools.product(
        models, methods, runs, ess_thresholds, n_particles_list, proposals
    )


def read_files(
    results_dir, models, methods, runs, ess_thresholds, n_particles_list, proposals
):
    all_experiment_data = []
    for args in _iter_args(
        models, methods, runs, ess_thresholds, n_particles_list, proposals
    ):
        all_experiment_data.extend(read_file(results_dir, *args))
    return all_experiment_data


def read_file(results_dir, model, method, run, ess, n_particles, proposal):
    fp = make_file_path(results_dir, model, method, ess, run, n_particles, proposal)
    if not os.path.exists(fp):
        warnings.warn(f'{fp} : could not find file')
        return []

    experiment_data = []
    with open(fp, 'r') as lines:
        for i, line in enumerate(lines):
            try:
                json_line = json.loads(line)
                json_line['model'] = model
                json_line['method'] = method
                json_line['run'] = run
                json_line['n_particles'] = n_particles
                json_line['proposal'] = proposal
                json_line['ess_threshold'] = ess
                if 'record' in json_line:
                    assert ess == json_line['record']['ess_threshold']
                    json_line['num_resample_steps'] = count_resample_steps(
                        json_line['record']
                    )
                else:
                    json_line['num_resample_steps'] = np.nan
                experiment_data.append(json_line)
            except json.JSONDecodeError as e:
                warnings.warn(f'{fp} : reading line {i} failed with {e}')

    return experiment_data


import pickle
import numpy as np
import pandas as pd
from scipy.stats import binom

with open('spider_question2level.pkl', 'rb') as f:
    question2level = pickle.load(f)


def make_experiment_df(data):
    columns = [
        'model',
        'method',
        'ess_threshold',
        'run',
        'n_particles',
        'proposal',
        'log_ml',
        'example_id',
        'num_resample_steps',
        'level',
    ]

    df = pd.DataFrame()

    for trial_data in data:
        df_line = [
            trial_data['model'],
            trial_data['method'],
            trial_data['ess_threshold'],
            trial_data['run'],
            trial_data['n_particles'],
            trial_data['proposal'],
            trial_data['log_ml'] if 'log_ml' in trial_data else None,
            trial_data['question'] + '-' + trial_data['db_name'],
            trial_data['num_resample_steps'],
            question2level[trial_data['question']],
        ]

        res_columns = []
        for res in trial_data['results']:
            results = trial_data['results'][res]
            res_columns.append(res)
            if results is None:
                df_line.append(np.nan)
            elif isinstance(results['result'], list) or isinstance(
                results['result'], tuple
            ):
                df_line.append(results['result'][0])
            elif isinstance(results['result'], float) or isinstance(
                results['result'], int
            ):
                df_line.append(results['result'])
            else:
                raise ValueError(results['result'])

        these_columns = columns + res_columns

        assert len(df_line) == len(these_columns), these_columns

        this_df = pd.DataFrame([df_line], columns=these_columns)

        df = pd.concat([df, this_df])

    return df
