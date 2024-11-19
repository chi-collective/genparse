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
from genparse.batch_inference import BatchVLLM, BatchStepModel, smc_steer
from genparse.batch_inference.steer import Particle, ParticleApproximation

from run_inference import (
    table_column_potential,
    strip_query_at_boundary,
    extract_latest_subquery,
    load_parser,
)

os.environ['TOKENIZERS_PARALLELISM'] = '(true|false)'

eps = 1e-10
log_eps = np.log(eps)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Spider inference with table column check potential using SMC steering baseline.'
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
        '--beam_range',
        type=int,
        nargs='+',
        default=[5],
        help='List of beam parameters to try.',
    )
    parser.add_argument(
        '--dev_data_min',
        type=int,
        default=0,
        help='Limit on the number of dev data points to process.',
    )
    parser.add_argument(
        '--dev_data_max',
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
                            instance['n_particles'],
                            instance['beam'],
                            instance['n_replicate'],
                        )
                    )
                except json.JSONDecodeError:
                    print(f'Error decoding line: {line}')
                    continue

    return processed_questions


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

        return smc_steer(step_model, potential=potential, **smc_args)

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

    def skip_instance(question, n_particles, beam, n_replicate):
        do_skip = (
            question,
            n_particles,
            beam,
            n_replicate,
        ) in processed_questions

        if do_skip:
            print(
                f'Skipping {question=} with {n_particles=} {n_replicate=} and' f'{beam=}'
            )

        return do_skip

    llm = vllm.LLM(
        model=args.model_name,
        rope_scaling={'type': 'dynamic', 'factor': 1.0},
        max_model_len=7760,
    )

    prompt_formatter = SpiderPromptFormatter(train_data, spider_schemas)

    f = open(args.output_file, 'a')

    print('Running GenParse with SMC steering')

    batch_llm = BatchVLLM(VirtualTokenizedLLM(llm.llm_engine))

    run_inference = spider_inference_setup(
        batch_llm=batch_llm,
        grammar_dir=args.grammar_dir,
        guide_cache_path=args.guide_cache_path,
        prompt_formatter=prompt_formatter,
    )

    for n_particles in args.n_particles_range:
        for n_replicate in range(args.n_replicates):
            for dev_datum in tqdm(dev_data[args.dev_data_min : args.dev_data_max]):
                potential = partial(
                    table_column_potential,
                    schema_name=dev_datum.schema_name,
                    grammar_path=args.grammar_dir,
                    spider_schemas=spider_schemas,
                    verbosity=0,
                )
                try:
                    for beam in args.beam_range:
                        if not skip_instance(
                            question=dev_datum.utterance,
                            n_particles=n_particles,
                            beam=beam,
                            n_replicate=n_replicate,
                        ):
                            particles = run_inference(
                                dev_datum,
                                potential=potential,
                                max_tokens=args.max_tokens,
                                n_particles=n_particles,
                                n_beam=beam,
                                return_record=True,
                            )

                            smc_results = {
                                'gold': dev_datum.query,
                                'db_name': dev_datum.schema_name,
                                'question': dev_datum.utterance,
                                'record': particles.record,
                                'results': {},
                                'n_particles': n_particles,
                                'method': 'smc_steer',
                                'beam': beam,
                                'n_replicate': n_replicate,
                            }

                            print(json.dumps(smc_results), file=f)

                except Exception as e:
                    f.close()
                    raise (e)
    f.close()


if __name__ == '__main__':
    main()
