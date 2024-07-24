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

from bench.spider.dialogue import load_spider_data
from bench.spider.evaluator import Evaluator
from bench.spider.schema import load_schemas
from bench.spider.prompt_formatter import SpiderPromptFormatter
from genparse.util import set_seed

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-query', type=int, default=100)
    parser.add_argument(
        '--model-name',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        choices=['meta-llama/Meta-Llama-3-8B-Instruct'],
    )
    parser.add_argument("--method", type=str, choices=["sampling", "greedy"],
                        default="sampling")
    parser.add_argument('--exp-name', type=str, default='llama3-8b')
    parser.add_argument('--particles', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)

    return parser


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    # for easy running from cli
    access_token = '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'
    os.environ['HF_TOKEN'] = access_token

    # Load data.
    logger.info('loading spider data...')
    raw_spider_dir = Path('bench/spider/data/spider')
    spider_schemas = load_schemas(
        schemas_path=raw_spider_dir / 'tables.json', db_path=raw_spider_dir / 'database'
    )  # Dict[str, DbSchema], schema_name -> schema

    spider_dev_data = load_spider_data(raw_spider_dir / 'dev.json')
    spider_train_data = load_spider_data(raw_spider_dir / 'train_spider.json')
    logger.info('spider data loaded.')

    prompt_formatter = SpiderPromptFormatter(spider_train_data, spider_schemas)
    evaluator = Evaluator(raw_spider_dir)

    # Initialize model.
    logger.info(f'Initializing model: {args.model_name} ...')
    llm = vllm.LLM(model=args.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    logger.info('Model(s) initialized.')

    # Prepare prompts.
    n_query = args.n_query
    prompts = []
    for dev_datum in tqdm(spider_dev_data[:n_query], desc="prompt"):
        messages = prompt_formatter.format_openai(dev_datum)
        # decode and later re-encode so that the prompt tokens are exactly the same as in
        # genparse.
        prompt = tokenizer.decode(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True))
        prompts.append(prompt)

    # Batch sampling.
    if args.method == "sampling":
        sampling_params = vllm.SamplingParams(
            n=args.particles, temperature=1.0, max_tokens=args.max_tokens, seed=args.seed)
    else:
        assert args.method == "greedy"
        sampling_params = vllm.SamplingParams(
            best_of=1, temperature=0.0, max_tokens=args.max_tokens)
    llm_outputs = llm.generate(prompts, sampling_params)

    # Process and write outputs.
    outpath = f"{args.exp_name}-{args.method}-p{args.particles}-{n_query}.jsonl"
    outfile = open(outpath, "w+")
    logger.info(f"writing to {outpath} ...")

    n_correct, n_invalid, n_mismatch = 0, 0, 0
    for dev_datum, output in zip(spider_dev_data, tqdm(llm_outputs, desc="output")):
        best = max(output.outputs, key=lambda x: x.cumulative_logprob)
        pred = best.text
        gold = dev_datum.query
        db = dev_datum.schema_name
        result = evaluator.evaluate(dev_datum.query, best.text, dev_datum.schema_name)

        if result[0]:
            n_correct += 1
        elif result[1] == "invalid":
            n_invalid += 1
        elif result[1] == "mismatch":
            n_mismatch += 1

        particles = [vars(o) for o in output.outputs]
        for p in particles:
            p["result"] = evaluator.evaluate(
                dev_datum.query, p["text"], dev_datum.schema_name)

        result_str = json.dumps({
            "pred": pred,
            "gold": gold,
            "db_name": db,
            "question": dev_datum.utterance,
            "result": result,
            "finished": best.finished(),
            "tokens": [tokenizer.decode(t) for t in best.token_ids],
            "token_ids": best.token_ids,
            "particles": [vars(o) for o in output.outputs],
        })

        print(result_str, file=outfile)

    n_total = sum((n_correct, n_invalid, n_mismatch))
    print(
        f'correct: {n_correct / n_total:.2f} ({n_correct}), '
        f'invalid: {n_invalid / n_total:.2f} ({n_invalid}), '
        f'mismatch: {n_mismatch / n_total:.2f} ({n_mismatch})'
    )


if __name__ == '__main__':
    main()