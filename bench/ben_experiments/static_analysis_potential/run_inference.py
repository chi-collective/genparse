import os
import json
import copy
import psutil
import argparse
from lark import Lark
from pathlib import Path

import vllm
import numpy as np
from tqdm import tqdm
from functools import partial

import bench.spider.schema
from bench.cache import ProposalCache
from bench.spider.schema import load_schemas
from bench.spider.dialogue import load_spider_data
from bench.spider.prompt_formatter import SpiderPromptFormatter, UtterancePromptFormatter

from genparse import EOS
from genparse.util import set_seed
from genparse.lm import VirtualTokenizedLLM
from genparse.batch_inference.lm import use_default_sampler
from genparse.batch_inference import BatchVLLM, BatchStepModel, smc
from genparse.batch_inference.steer import Particle, ParticleApproximation

from functools import lru_cache
from table_column_verifier import ColumnValidator

os.environ['TOKENIZERS_PARALLELISM'] = '(true|false)'

eps = 1e-10
log_eps = np.log(eps)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Spider inference with utterance potential.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Meta-Llama-3.1-8B-Instruct',
        help='Name of the model to use.',
    )
    parser.add_argument(
        '--raw_spider_dir',
        type=str,
        default='../../spider/data/spider',
        help='Path to the raw Spider data directory.',
    )
    parser.add_argument(
        '--max_tokens', type=int, default=100, help='Maximum number of tokens.'
    )
    parser.add_argument(
        '--grammar_dir',
        type=str,
        default='../spider_grammars',
        help='Path to the Spider grammar directory.',
    )
    parser.add_argument(
        '--guide_cache_path',
        type=str,
        default='guide_cache.pkl',
        help='Path to the guide cache file.',
    )
    parser.add_argument(
        '--n_particles_range',
        type=int,
        nargs='+',
        default=[10],
        help='Number of particles for SMC/SIS methods.',
    )
    parser.add_argument(
        '--ess_threshold_range',
        type=float,
        nargs='+',
        default=[0.9],
        help='List ESS thresholds for resampling to try.',
    )
    parser.add_argument(
        '--resample_method',
        type=str,
        default='multinomial',
        help='Resampling method to use.',
    )
    parser.add_argument(
        '--dev_data_limit',
        type=int,
        default=1034,
        help='Limit on the number of dev data points to process.',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='File to write the inference results.',
    )
    parser.add_argument(
        '--n_replicates',
        type=int,
        default=1,
        help='Number of replicates to run for each datum.',
    )
    parser.add_argument(
        '--run_baseline',
        action='store_true',
        help='Run baseline inference.',
    )
    parser.add_argument(
        '--run_genparse',
        action='store_true',
        help='Run GenParse inference.',
    )

    return parser.parse_args()


def load_existing_results(output_file):
    processed_questions = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    instance = json.loads(line.strip())
                    processed_questions.add(
                        (
                            instance['question'],
                            instance['method'],
                            instance['ess_threshold']
                            if 'ess_threshold' in instance
                            else None,
                            instance['resample_method']
                            if 'resample_method' in instance
                            else None,
                            instance['n_replicate'],
                        )
                    )
                except json.JSONDecodeError:
                    print(f'Error decoding line: {line}')
                    continue

    return processed_questions


@lru_cache
def load_parser(schema_name, grammar_path):
    parser = Lark(open(os.path.join(grammar_path, schema_name + '.lark'), 'r').read())

    @lru_cache
    def parse(x):
        return parser.parse(x)

    return parse


import re


def extract_latest_subquery(query):
    # capture the latest SELECT statement within parentheses (partial or complete)
    subquery_pattern = r'\(\s*(SELECT.*?)(?:\s*\)|$)'  # Matches until closing parenthesis or end of string
    subqueries = re.findall(subquery_pattern, query, re.IGNORECASE | re.DOTALL)
    return subqueries[-1] if subqueries else None


def strip_query_at_boundary(query):
    if query.endswith(EOS):
        return query.rstrip(EOS)
    elif query.endswith('WHERE'):
        return query.rstrip('WHERE')
    elif query.endswith('GROUP BY'):
        return query.rstrip('GROUP BY')
    elif query.endswith('ORDER'):
        return query.rstrip('ORDER')
    else:
        return None


def table_column_potential(
    particles, schema_name, grammar_path, spider_schemas, verbosity=0, **kwargs
):
    parser = load_parser(schema_name, grammar_path)
    tables = spider_schemas[schema_name].tables

    potential_values = []
    for p in particles:
        query = strip_query_at_boundary(''.join(p.context))

        if (query is None) and (not p.done):
            potential_values.append(0)
            continue

        try:
            parse = parser(query)
        except Exception:
            try:
                subquery = extract_latest_subquery(query)
                if subquery is None:
                    potential_values.append(0)
                    continue

                subquery = strip_query_at_boundary(subquery)
                if subquery is None:
                    potential_values.append(0)
                    continue

                parse = parser(subquery)
            except Exception as e:
                print('Failed to parse:', query, e)
                potential_values.append(0)
                continue

        validator = ColumnValidator(tables, verbosity)
        try:
            validator.transform(parse)
            value = 0 if validator.is_valid else log_eps
            potential_values.append(value)
        except Exception as e:
            print('Failed to validate:', query, e)
            potential_values.append(0)

    return potential_values


def spider_inference_setup(
    batch_llm, grammar_dir, prompt_formatter, **proposal_cache_args
):
    proposal_cache = ProposalCache(**proposal_cache_args)
    tokenizer = batch_llm.llm.tokenizer

    def run_inference(datum, potential=None, max_tokens=150, **smc_args):
        mem_usage = psutil.virtual_memory().percent
        if mem_usage > 60:
            print(f'Memory usage: {mem_usage}%. Clearing cache.')
            proposal_cache.clear_cache()
            print(f'New memory usage: {psutil.virtual_memory().percent}%')

        grammar = open(os.path.join(grammar_dir, datum.schema_name + '.lark'), 'r').read()

        parallel_proposal = proposal_cache.fetch_or_create_proposal(
            llm=batch_llm.llm,
            grammar=grammar,
            proposal_name='character',
            n_processes=10,
            proposal_args={},
        )

        step_model = BatchStepModel(
            batch_proposal=parallel_proposal,
            batch_llm=batch_llm,
            max_tokens=max_tokens,
        )

        step_model.set_prompt(
            tokenizer.apply_chat_template(
                prompt_formatter.format_openai(datum),
                add_generation_prompt=True,
                tokenize=False,
            )
        )

        return smc(step_model, potential=potential, **smc_args)

    return run_inference


def main():
    args = parse_args()
    set_seed(0)

    raw_spider_dir = Path(args.raw_spider_dir)
    train_data = load_spider_data(raw_spider_dir / 'train_spider.json')
    dev_data = load_spider_data(raw_spider_dir / 'dev.json')
    spider_schemas = load_schemas(
        schemas_path=raw_spider_dir / 'tables.json', db_path=raw_spider_dir / 'database'
    )

    processed_questions = load_existing_results(args.output_file)

    def skip_instance(question, method, ess_threshold, resample_method, n_replicate):
        do_skip = (
            question,
            method,
            ess_threshold,
            resample_method,
            n_replicate,
        ) in processed_questions

        if do_skip:
            print(
                f'Skipping {question=} with {method=}, {n_replicate=} '
                f'{ess_threshold=} and {resample_method=} '
            )

        return do_skip

    llm = vllm.LLM(
        model=args.model_name,
        rope_scaling={'type': 'dynamic', 'factor': 1.0},
        max_model_len=7760,
    )

    tokenizer = llm.get_tokenizer()

    prompt_formatter = SpiderPromptFormatter(train_data, spider_schemas)

    f = open(args.output_file, 'a')

    if args.run_baseline:
        print('Running baseline')

        prompts = [
            tokenizer.apply_chat_template(
                prompt_formatter.format_openai(datum),
                add_generation_prompt=True,
                tokenize=False,
            )
            for datum in dev_data[: args.dev_data_limit]
        ]

        with use_default_sampler(llm):
            for n_particles in args.n_particles_range:
                for n_replicate in range(args.n_replicates):
                    llm_outputs = llm.generate(
                        prompts,
                        vllm.SamplingParams(
                            n=n_particles, temperature=1.0, max_tokens=150, seed=0
                        ),
                    )

                    for dev_datum, output in zip(
                        dev_data[: args.dev_data_limit], tqdm(llm_outputs)
                    ):
                        particles = ParticleApproximation(
                            [
                                Particle(
                                    prompt=None,
                                    context=out.text,
                                    context_ids=out.token_ids,
                                    done=out.finish_reason == 'stop',
                                    log_weight=0,
                                    parent=None,
                                )
                                for out in output.outputs
                            ]
                        )

                        baseline_results = {
                            'gold': dev_datum.query,
                            'db_name': dev_datum.schema_name,
                            'question': dev_datum.utterance,
                            'record': {
                                'history': [
                                    {
                                        'particles': [
                                            {
                                                'context': p.context,
                                                'context_ids': p.context_ids,
                                                'weight': p.log_weight,
                                                'finished': p.done,
                                            }
                                            for p in particles
                                        ]
                                    }
                                ]
                            },
                            'results': {},
                            'n_particles': n_particles,
                            'method': 'lm_baseline',
                            'ess_threshold': None,
                            'resample_method': None,
                            'n_replicate': n_replicate,
                        }

                        print(json.dumps(baseline_results), file=f)

    if not args.run_genparse:
        f.close()
        return

    print('Running GenParse')

    batch_llm = BatchVLLM(VirtualTokenizedLLM(llm.llm_engine))

    run_inference = spider_inference_setup(
        batch_llm=batch_llm,
        grammar_dir=args.grammar_dir,
        guide_cache_path=args.guide_cache_path,
        prompt_formatter=prompt_formatter,
    )

    for n_particles in args.n_particles_range:
        for n_replicate in range(args.n_replicates):
            for dev_datum in tqdm(dev_data[: args.dev_data_limit]):
                potential = partial(
                    table_column_potential,
                    schema_name=dev_datum.schema_name,
                    grammar_path=args.grammar_dir,
                    spider_schemas=spider_schemas,
                    verbosity=0,
                )
                try:
                    for ess_threshold in args.ess_threshold_range:
                        if not skip_instance(
                            question=dev_datum.utterance,
                            method='smc',
                            ess_threshold=ess_threshold,
                            resample_method=args.resample_method,
                            n_replicate=n_replicate,
                        ):
                            particles = run_inference(
                                dev_datum,
                                potential=potential,
                                max_tokens=args.max_tokens,
                                n_particles=n_particles,
                                ess_threshold=ess_threshold,
                                return_record=True,
                                resample_method=args.resample_method,
                            )

                            smc_results = {
                                'gold': dev_datum.query,
                                'db_name': dev_datum.schema_name,
                                'question': dev_datum.utterance,
                                'record': particles.record,
                                'results': {},
                                'n_particles': n_particles,
                                'method': 'smc',
                                'ess_threshold': ess_threshold,
                                'n_replicate': n_replicate,
                                'resample_method': args.resample_method,
                            }

                            print(json.dumps(smc_results), file=f)

                        if (
                            not skip_instance(
                                question=dev_datum.utterance,
                                method='smc_no_potential',
                                ess_threshold=ess_threshold,
                                resample_method=args.resample_method,
                                n_replicate=n_replicate,
                            )
                            and False
                        ):  # XXX: Disabled for now
                            particles = run_inference(
                                dev_datum,
                                potential=None,
                                max_tokens=args.max_tokens,
                                n_particles=n_particles,
                                ess_threshold=ess_threshold,
                                return_record=True,
                                resample_method=args.resample_method,
                            )

                            smc_results = {
                                'gold': dev_datum.query,
                                'db_name': dev_datum.schema_name,
                                'question': dev_datum.utterance,
                                'record': particles.record,
                                'results': {},
                                'n_particles': n_particles,
                                'method': 'smc_no_potential',
                                'ess_threshold': ess_threshold,
                                'n_replicate': n_replicate,
                                'resample_method': args.resample_method,
                            }

                            print(json.dumps(smc_results), file=f)

                    if (
                        skip_instance(
                            question=dev_datum.utterance,
                            method='sis',
                            ess_threshold=None,
                            resample_method=None,
                            n_replicate=n_replicate,
                        )
                        and skip_instance(
                            question=dev_datum.utterance,
                            method='sis_no_potential',
                            ess_threshold=None,
                            resample_method=None,
                            n_replicate=n_replicate,
                        )
                        and True
                    ):  # XXX: Disabled for now
                        continue

                    particles = run_inference(
                        dev_datum,
                        potential=potential,
                        max_tokens=args.max_tokens,
                        n_particles=n_particles,
                        ess_threshold=0,
                        return_record=True,
                    )

                    sis_no_potential_results = {
                        'gold': dev_datum.query,
                        'db_name': dev_datum.schema_name,
                        'question': dev_datum.utterance,
                        'record': particles.record,
                        'results': {},
                        'n_particles': n_particles,
                        'method': 'sis_no_potential',
                        'resample_method': None,
                        'n_replicate': n_replicate,
                    }

                    print(json.dumps(sis_no_potential_results), file=f)

                    record = copy.deepcopy(particles.record)

                    record['history'][-1]['particles'] = [
                        {
                            'context': p.context,
                            'weight': p.log_weight,
                            'context_ids': p.context_ids,
                        }
                        for p in particles.particles  # add potential particles
                    ]

                    sis_results = {
                        'gold': dev_datum.query,
                        'db_name': dev_datum.schema_name,
                        'question': dev_datum.utterance,
                        'record': record,
                        'results': {},
                        'n_particles': n_particles,
                        'method': 'sis',
                        'resample_method': None,
                        'n_replicate': n_replicate,
                    }

                    print(json.dumps(sis_results), file=f)

                except Exception as e:
                    print(e)
                    continue
    f.close()


if __name__ == '__main__':
    main()
