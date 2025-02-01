#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate SMCalFlow on vLLM Llama-3 (baseline).

Example usage:
python bench/run_spider_baseline.py --n-query 1000 --method greedy --exp-name llama-3-baseline
"""

import argparse
import logging
import os
import pprint
from pathlib import Path

import vllm
from tqdm import tqdm

from bench.calflow import is_correct
from bench.calflow.datum import data_from_filename, transform_datum
from bench.calflow.bm25_index import BM25Retriever
from bench.calflow.fewshot import PromptBuilder
from genparse.util import set_seed

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-query', type=int, default=100)
    parser.add_argument(
        '--model-name',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
    )
    parser.add_argument(
        '--method', type=str, choices=['sampling', 'greedy'], default='sampling'
    )
    parser.add_argument('--exp-name', type=str, default='llama3-8b')
    parser.add_argument('--particles', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=300)
    parser.add_argument('--k-shot', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--train-file',
        type=str,
        default='train_low_0.jsonl',
        help='train split file name from bench clamp',
    )
    parser.add_argument(
        '--dev-file',
        type=str,
        default='dev_low.jsonl',
        help='dev split file name from bench clamp',
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
    set_seed(args.seed)

    logger.info('loading data...')
    calflow_data_root = Path(
        '/home/leodu/semantic_parsing_with_constrained_lm/data/benchclamp/processed/CalFlowV2'
    )
    train_data_file = calflow_data_root / args.train_file
    dev_data_file = calflow_data_root / args.dev_file

    train_data = transform_datum(data_from_filename(train_data_file))
    dev_data = transform_datum(data_from_filename(dev_data_file))
    logger.info('finished loading data.')

    logger.info(f'Example datum:\n {pprint.pformat(train_data[0])}')

    train_retriever = BM25Retriever(
        train_data=train_data, top_k=args.k_shot, best_first=False
    )
    prompt_builder = PromptBuilder.for_demo(
        do_include_context=False, use_preamble=True
    )  # configs.lib.common:83

    access_token = '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'
    os.environ['HF_TOKEN'] = access_token

    if args.method == 'sampling':
        sampling_params = vllm.SamplingParams(
            n=args.particles,
            temperature=1.0,
            max_tokens=args.max_tokens,
            seed=args.seed,
            stop='\n',
        )
    else:
        assert args.method == 'greedy'
        sampling_params = vllm.SamplingParams(
            best_of=1, temperature=0.0, max_tokens=args.max_tokens, stop='\n'
        )

    # Prepare the prompts and collect them in `samples`
    samples = []
    for dev_datum in tqdm(dev_data):
        selected_train_data = train_retriever(dev_datum)
        prompt = prompt_builder.assemble(selected_train_data, dev_datum)
        samples.append({'datum': dev_datum, 'prompt': prompt})

    logger.info(f'Example prompt:\n{samples[0]["prompt"]}')

    llm = vllm.LLM(model=args.model_name)
    llm_outputs = llm.generate([s['prompt'] for s in samples], sampling_params)

    nc = 0
    ntotal = 0
    for llm_output, sample in zip(llm_outputs, tqdm(samples)):
        ntotal += 1
        pred = llm_output.outputs[0].text
        nc += is_correct(pred, sample['datum'])
    logger.info(f' {nc} out of {ntotal} is correct, acc={nc / ntotal:.3f}')


if __name__ == '__main__':
    main()
