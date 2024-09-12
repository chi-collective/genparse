#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################
# Runs VLLM baseline on spider dataset #
########################################

import os
import json
import logging
import argparse
from pathlib import Path

import vllm
import transformers
from tqdm import tqdm

from genparse.batch_inference.steer import Particle, ParticleApproximation

from bench.spider.evaluator import Evaluator
from utils import (
    load_spider_data,
    load_prompt_formatter,
    HF_ACCESS_TOKEN,
    posterior_weighted_eval,
    mbr_eval,
)
from genparse.util import set_seed

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-name',
        type=str,
    )
    parser.add_argument(
        '--method', type=str, choices=['sampling', 'greedy'], default='sampling'
    )
    parser.add_argument('--exp-name', type=str, default='llama3-8b')
    parser.add_argument('--particles', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='')
    parser.add_argument(
        '--schema',
        type=str,
        help='Schema to evaluate, seperated by commas. Defaults to `all` for all schema.',
        default='all',
    )

    return parser


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )
    parser = get_parser()
    args = parser.parse_args()

    outpath = os.path.join(
        args.out_dir, f'{args.exp_name}-p{args.particles}-{args.method}'
    )

    json.dump(vars(args), open(f'{outpath}-args.json', 'w'), indent=4)

    outpath += '.jsonl'
    outfile = open(outpath, 'w+')

    set_seed(args.seed)

    raw_spider_dir = Path('../../spider/data/spider')
    spider_dev_data = load_spider_data(raw_spider_dir, split='dev')
    evaluator = Evaluator(raw_spider_dir)
    prompt_formatter = load_prompt_formatter(raw_spider_dir)

    os.environ['HF_TOKEN'] = HF_ACCESS_TOKEN

    # Initialize model.
    logger.info(f'Initializing model: {args.model_name} ...')
    if '3.1' in args.model_name:
        llm = vllm.LLM(
            model=args.model_name,
            rope_scaling={'type': 'dynamic', 'factor': 8.0},
            max_model_len=7760,
        )
    else:
        llm = vllm.LLM(model=args.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    logger.info('Model(s) initialized.')

    if args.schema == 'all':
        schema = list(set([d.schema_name for d in spider_dev_data]))
        print(f'Using schema {",".join(schema)}')
    else:
        schema = args.schema.split(',')

    # Prepare prompts.
    prompts = []
    dev_data = []
    for dev_datum in tqdm(spider_dev_data, desc='prompt'):
        if dev_datum.schema_name not in schema:
            continue
        dev_data.append(dev_datum)
        messages = prompt_formatter.format_openai(dev_datum)
        # decode and later re-encode so that the prompt tokens are exactly the same as in
        # genparse.
        prompt = tokenizer.decode(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        )
        prompts.append(prompt)

    # Batch sampling.
    if args.method == 'sampling':
        sampling_params = vllm.SamplingParams(
            n=args.particles, temperature=1.0, max_tokens=args.max_tokens, seed=args.seed
        )
    else:
        assert args.method == 'greedy'
        sampling_params = vllm.SamplingParams(
            best_of=1, temperature=0.0, max_tokens=args.max_tokens
        )

    llm_outputs = llm.generate(prompts, sampling_params)

    assert len(dev_data) == len(llm_outputs)

    n_correct, n_invalid, n_mismatch = 0, 0, 0
    for dev_datum, output in zip(dev_data, tqdm(llm_outputs, desc='output')):
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

        particles_json = [
            {
                'tokens': p.context,
                'token_ids': p.context_ids,
                'weight': p.log_weight,
                'finished': p.done,
            }
            for p in particles
        ]

        gold = dev_datum.query
        db = dev_datum.schema_name

        json_result = {
            'gold': gold,
            'db_name': db,
            'question': dev_datum.utterance,
            'particles': particles_json,
            'log_ml': None,
            'results': {},
        }

        json_result['results']['mbr'] = mbr_eval(particles, evaluator, gold, db, '')
        json_result['results']['posterior_weighted_acc'] = posterior_weighted_eval(
            particles, evaluator, gold, db, ''
        )

        viterbi_best = max(output.outputs, key=lambda x: x.cumulative_logprob)
        pred = viterbi_best.text
        json_result['results']['viterbi'] = {
            'result': evaluator.evaluate(gold, pred, db),
            'pred': pred,
            'finished': viterbi_best.finish_reason == 'stop',
            'token_ids': viterbi_best.token_ids,
        }

        result = json_result['results']['mbr']['result']

        if result[0]:
            n_correct += 1
        elif result[1] == 'invalid':
            n_invalid += 1
        elif result[1] == 'mismatch':
            n_mismatch += 1

        result_str = json.dumps(json_result)

        print(result_str, file=outfile)

    n_total = sum((n_correct, n_invalid, n_mismatch))
    print(
        f'correct: {n_correct / n_total:.2f} ({n_correct}), '
        f'invalid: {n_invalid / n_total:.2f} ({n_invalid}), '
        f'mismatch: {n_mismatch / n_total:.2f} ({n_mismatch})'
    )


if __name__ == '__main__':
    main()
