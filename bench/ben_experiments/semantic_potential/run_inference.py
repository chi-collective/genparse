# Example usage: CUDA_VISIBLE_DEVICES=1 python run_inference.py --temperature_range 1.5 1.0 1.25 1.75 2.0 --output_file temperature_results.jsonl

import os
import json
import copy
import psutil
import argparse
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
from genparse.batch_inference import BatchVLLM, BatchStepModel, smc

from potential import utterance_potential, CachedScorer, utterance_prompter

os.environ['TOKENIZERS_PARALLELISM'] = '(true|false)'


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
        '--burn_in_range', type=int, nargs='+', help='List of burn ins to try.'
    )
    parser.add_argument(
        '--n_particles',
        type=int,
        default=10,
        help='Number of particles for SMC/SIS methods.',
    )
    parser.add_argument(
        '--ess_threshold', type=float, default=0.5, help='ESS threshold for resampling.'
    )
    parser.add_argument(
        '--resample_method',
        type=str,
        default='multinomial',
        help='Resampling method to use.',
    )
    parser.add_argument(
        '--temperature_range', type=float, nargs='+', help='List of temperatures to try.'
    )
    parser.add_argument(
        '--dev_data_limit',
        type=int,
        default=100,
        help='Limit on the number of dev data points to process.',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='temperature_results.jsonl',
        help='File to write the inference results.',
    )

    return parser.parse_args()


def load_existing_results(output_file):
    processed_questions = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    if result['burn_in'] not in processed_questions:
                        processed_questions[result['burn_in']] = {}
                    if (
                        result['temperature']
                        not in processed_questions[result['burn_in']]
                    ):
                        processed_questions[result['burn_in']][result['temperature']] = (
                            set()
                        )
                    processed_questions[result['burn_in']][result['temperature']].add(
                        result['question']
                    )
                except json.JSONDecodeError:
                    print(f'Error decoding line: {line}')
                    continue
    return processed_questions


class SpiderInferenceSetup:
    def __init__(self, batch_llm, grammar_dir, prompt_formatter, **proposal_cache_args):
        self.batch_llm = batch_llm
        self.tokenizer = self.batch_llm.llm.tokenizer
        self.grammar_dir = grammar_dir
        self.proposal_cache = ProposalCache(**proposal_cache_args)
        self.prompt_formatter = prompt_formatter

    def __call__(self, datum, potential=None, max_tokens=100, **smc_args):
        mem_usage = psutil.virtual_memory().percent
        if mem_usage > 60:
            print(f'Memory usage: {mem_usage}%. Clearing cache.')
            self.proposal_cache.clear_cache()
            print(f'New memory usage: {psutil.virtual_memory().percent}%')

        grammar = open(
            os.path.join(self.grammar_dir, datum.schema_name + '.lark'), 'r'
        ).read()

        parallel_proposal = self.proposal_cache.fetch_or_create_proposal(
            llm=self.batch_llm.llm,
            grammar=grammar,
            proposal_name='character',
            n_processes=10,
            proposal_args={},
        )

        step_model = BatchStepModel(
            batch_proposal=parallel_proposal,
            batch_llm=self.batch_llm,
            max_tokens=max_tokens,
        )

        step_model.set_prompt(
            self.tokenizer.apply_chat_template(
                self.prompt_formatter.format_openai(datum),
                add_generation_prompt=True,
                tokenize=False,
            )
        )

        return smc(step_model, potential=potential, **smc_args)


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

    llm = vllm.LLM(
        model=args.model_name,
        rope_scaling={'type': 'dynamic', 'factor': 1.0},
        max_model_len=7760,
    )

    tokenizer = llm.get_tokenizer()
    batch_llm = BatchVLLM(VirtualTokenizedLLM(llm.llm_engine))
    prompt_formatter = SpiderPromptFormatter(train_data, spider_schemas)
    run_inference = SpiderInferenceSetup(
        batch_llm=batch_llm,
        grammar_dir=args.grammar_dir,
        guide_cache_path=args.guide_cache_path,
        prompt_formatter=prompt_formatter,
    )

    utterance_prompt_formatter = partial(
        utterance_prompter, spider_train_data=train_data, tokenizer=tokenizer
    )
    scorer = CachedScorer(batch_llm, cache_path='scorer_cache.pkl')

    f = open(args.output_file, 'a')

    for burn_in in args.burn_in_range:
        for temperature in args.temperature_range:
            for dev_datum in tqdm(dev_data[: args.dev_data_limit]):
                if burn_in in processed_questions:
                    if temperature in processed_questions[burn_in]:
                        if (
                            dev_datum.utterance
                            in processed_questions[burn_in][temperature]
                        ):
                            print(
                                f'Skipping: `{dev_datum.utterance}`'
                                f' with temperature: {temperature} and burn-in: {burn_in}'
                            )
                            continue

                potential = partial(
                    utterance_potential,
                    scorer=scorer,
                    prompt_formatter=utterance_prompt_formatter,
                    temperature_schedule=lambda t: np.inf if t < burn_in else temperature,
                    utterance_ids=tokenizer.encode(
                        dev_datum.utterance, add_special_tokens=False
                    ),
                )

                try:
                    particles = run_inference(
                        dev_datum,
                        potential=potential,
                        max_tokens=args.max_tokens,
                        n_particles=args.n_particles,
                        ess_threshold=args.ess_threshold,
                        return_record=True,
                        resample_method=args.resample_method,
                    )

                    smc_results = {
                        'gold': dev_datum.query,
                        'db_name': dev_datum.schema_name,
                        'question': dev_datum.utterance,
                        'record': particles.record,
                        'results': {},
                        'n_particles': args.n_particles,
                        'method': 'smc',
                        'temperature': temperature,
                        'burn_in': burn_in,
                        'ess_threshold': args.ess_threshold,
                    }

                    print(json.dumps(smc_results), file=f)

                    particles = run_inference(
                        dev_datum,
                        potential=potential,
                        max_tokens=args.max_tokens,
                        n_particles=args.n_particles,
                        ess_threshold=0,
                        return_record=True,
                    )

                    sis_no_potential_results = {
                        'gold': dev_datum.query,
                        'db_name': dev_datum.schema_name,
                        'question': dev_datum.utterance,
                        'record': particles.record,
                        'results': {},
                        'n_particles': args.n_particles,
                        'method': 'sis_no_potential',
                        'temperature': temperature,
                        'burn_in': burn_in,
                        'ess_threshold': args.ess_threshold,
                    }

                    print(json.dumps(sis_no_potential_results), file=f)

                    record = copy.deepcopy(particles.record)

                    record['history'][-1]['particles'] = [
                        {
                            'context': p.context,
                            'weight': p.log_weight,
                            'context_ids': p.context_ids,
                        }
                        for p in particles.particles
                    ]

                    sis_results = {
                        'gold': dev_datum.query,
                        'db_name': dev_datum.schema_name,
                        'question': dev_datum.utterance,
                        'record': record,
                        'results': {},
                        'n_particles': args.n_particles,
                        'method': 'sis',
                        'temperature': temperature,
                        'burn_in': burn_in,
                        'ess_threshold': args.ess_threshold,
                    }

                    print(json.dumps(sis_results), file=f)

                except Exception as e:
                    print(e)
                    continue

    scorer.save_cache()
    f.close()


if __name__ == '__main__':
    main()
