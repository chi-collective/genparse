#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate spider on llama2-chat models without any grammar restriction."""

import argparse
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from bench.spider.dialogue import load_spider_data
from bench.spider.schema import load_schemas
from bench.spider.prompt_formatter import SpiderPromptFormatter

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-query', type=int, default=100)
    parser.add_argument('--model-size', type=str, default='7b', choices=['7b', '13b'])
    parser.add_argument('--exp-name', type=str, default='7b-100')

    return parser


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(0)

    raw_spider_dir = Path('bench/spider/data/spider')
    spider_schemas = load_schemas(
        schemas_path=raw_spider_dir / 'tables.json', db_path=raw_spider_dir / 'database'
    )

    spider_dev_data = load_spider_data(raw_spider_dir / 'dev.json')
    spider_train_data = load_spider_data(raw_spider_dir / 'train_spider.json')
    prompt_formatter = SpiderPromptFormatter(spider_train_data, spider_schemas)

    model = 'meta-llama/Llama-2-7b-chat-hf'
    model = model.replace('7b', args.model_size)
    access_token = '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'
    logger.info(f'using model {model}')

    # tokenizer = AutoTokenizer.from_pretrained(model, token=access_token)
    pipe = pipeline(
        'text-generation',
        model=model,
        model_kwargs={'load_in_8bit': True},
        torch_dtype=torch.float16,
        device_map='auto',
        token=access_token,
    )

    n_query = args.n_query

    gold = []
    predicted = []

    for i, dev_datum in tqdm(enumerate(spider_dev_data[:n_query]), total=n_query):
        prompt = prompt_formatter.format_llama2(dev_datum)
        if i == 0:
            print('=' * 30 + ' Example prompt ' + '=' * 30)
            print(prompt)
            print('=' * 30 + '  End of prompt ' + '=' * 30)
        output = pipe(prompt, do_sample=False, top_p=None, temperature=None)
        gold.append(dev_datum)
        predicted.append(output[0]['generated_text'][len(prompt) :])

    gold = spider_dev_data[:n_query]
    with open(f'bench/spider-eval/gold-{args.exp_name}.txt', 'w+') as f:
        for datum in gold:
            print(f'{datum.query}\t{datum.schema_name}', file=f)

    with open(f'bench/spider-eval/predicted-{args.exp_name}.txt', 'w+') as f:
        for datum in predicted:
            datum = datum.replace('\n', ' ')
            assert '\t' not in datum
            print(datum.strip(), file=f)


if __name__ == '__main__':
    main()
