#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate spider on vLLM Llama models without any grammar restriction."""

import argparse
import logging
import os
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from bench.spider.dialogue import load_spider_data
from bench.spider.schema import load_schemas
from bench.spider.prompt_formatter import SpiderPromptFormatter

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
    parser.add_argument('--exp-name', type=str, default='llama3-8b-100')
    parser.add_argument('--api-base', type=str, default='http://localhost:9999/v1')
    parser.add_argument('--api-key', type=str, default='EMPTY')

    return parser


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )
    # disable unnecessary logs from httpx used by openai client
    logging.getLogger('httpx').setLevel(logging.WARNING)
    parser = get_parser()
    args = parser.parse_args()

    logger.info('loading spider data...')
    raw_spider_dir = Path('bench/spider/data/spider')
    spider_schemas = load_schemas(
        schemas_path=raw_spider_dir / 'tables.json', db_path=raw_spider_dir / 'database'
    )

    spider_dev_data = load_spider_data(raw_spider_dir / 'dev.json')
    spider_train_data = load_spider_data(raw_spider_dir / 'train_spider.json')
    logger.info('spider data loaded.')

    prompt_formatter = SpiderPromptFormatter(spider_train_data, spider_schemas)

    logger.info(f"Creating client for '{args.model_name}' served at {args.api_base}")
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )
    n_query = args.n_query

    gold = []
    predicted = []

    for i, dev_datum in tqdm(enumerate(spider_dev_data[:n_query]), total=n_query):
        messages = prompt_formatter.format_openai(dev_datum)

        if i == 0:  # print an example for demonstration
            print('=' * 30 + ' Example prompt ' + '=' * 30)
            for msg in messages:
                print(msg['role'] + ':')
                print('=' * (len(msg['role']) + 1))
                print(msg['content'])
                print('-' * 100)
            print('=' * 30 + '  End of prompt ' + '=' * 30)

        chat_response = client.chat.completions.create(
            model=args.model_name,
            # model="mistralai/Mistral-7B-Instruct-v0.1",
            messages=messages,
            seed=0,
        )
        gold.append(dev_datum)
        predicted.append(chat_response.choices[0].message.content)

    gold = spider_dev_data[:n_query]

    gold_outfile = f'bench/spider-eval/gold-{args.exp_name}.txt'
    pred_outfile = f'bench/spider-eval/predicted-{args.exp_name}.txt'

    logger.info(f'saving output to {gold_outfile} and {pred_outfile}')

    with open(gold_outfile, 'w+') as f:
        for datum in gold:
            print(f'{datum.query}\t{datum.schema_name}', file=f)

    with open(pred_outfile, 'w+') as f:
        for datum in predicted:
            datum = datum.replace('\n', ' ')
            assert '\t' not in datum
            print(datum.strip(), file=f)


if __name__ == '__main__':
    main()
