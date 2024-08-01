#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate spider on vLLM Llama-3 (baseline).

Example usage:
python bench/run_spider_baseline.py --n-query 1000 --method greedy --exp-name llama-3-baseline
"""

import argparse
import json
import logging
import os
from pathlib import Path

import transformers
import vllm
from tqdm import tqdm

from genparse.experimental.batch_inference.steer import Particle, ParticleApproximation

from bench.spider.evaluator import Evaluator
from utils import load_spider_data, load_prompt_formatter, HF_ACCESS_TOKEN
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
        '--schema', type=str, help='Schema to evaluat, seperated by comma'
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
        best = max(output.outputs, key=lambda x: x.cumulative_logprob)

        def match(x, y):
            try:
                (exec_match, _) = evaluator.evaluate(x, y, db_name=dev_datum.schema_name)
            except Exception:
                exec_match = False
            return exec_match

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

        pmax = max(
            particles,
            key=lambda candidate: particles.risk(match, candidate.context),
        )

        pred = pmax
        gold = dev_datum.query
        db = dev_datum.schema_name
        result = evaluator.evaluate(dev_datum.query, best.text, dev_datum.schema_name)

        result_json = {
            'pred': pred,
            'gold': gold,
            'db_name': db,
            'question': dev_datum.utterance,
            'result': result,
            'finished': best.finished(),
            'tokens': [tokenizer.decode(t) for t in best.token_ids],
            'token_ids': best.token_ids,
            'particles': [vars(o) for o in output.outputs],
            'mbr': result,
        }

        if result[0]:
            n_correct += 1
        elif result[1] == 'invalid':
            n_invalid += 1
        elif result[1] == 'mismatch':
            n_mismatch += 1

        particles = [vars(o) for o in output.outputs]
        for p in particles:
            p['result'] = evaluator.evaluate(
                dev_datum.query, p['text'], dev_datum.schema_name
            )

        result_str = json.dumps(result_json)

        print(result_str, file=outfile)

    n_total = sum((n_correct, n_invalid, n_mismatch))
    print(
        f'correct: {n_correct / n_total:.2f} ({n_correct}), '
        f'invalid: {n_invalid / n_total:.2f} ({n_invalid}), '
        f'mismatch: {n_mismatch / n_total:.2f} ({n_mismatch})'
    )

    print(
        json.dumps(
            {
                'correct': n_correct,
                'invalid': n_invalid,
                'mismatch': n_mismatch,
                'n_total': n_total,
            }
        ),
        file=outfile,
    )


if __name__ == '__main__':
    main()
