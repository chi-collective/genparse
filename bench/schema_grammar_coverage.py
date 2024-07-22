#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test parser coverage on spider dev set using schema-specific grammars.

Note that parsers are constructed asynchronously to save time.

Example usage:
$ python bench/schema_grammar_coverage.py
"""

import json
import logging
import multiprocessing as mp
import os
from pathlib import Path

from tqdm import tqdm

from genparse.util import lark_guide
from bench.spider.dialogue import load_spider_data
from bench.spider.schema import load_schemas

logger = logging.getLogger(__name__)


def reformat_grammar(grammar):
    """move start rule and remove zero-width rules"""
    lines = grammar.split('\n')
    new_grammar = ''
    for line in lines:
        if line == '|""i':
            continue
        if line.startswith('start'):
            new_grammar = line + '\n' + new_grammar
        else:
            new_grammar += line + '\n'

    return new_grammar


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )

    # Load data.
    logger.info('loading spider dev data...')
    raw_spider_dir = Path('bench/spider/data/spider')

    spider_dev_data = load_spider_data(raw_spider_dir / 'dev.json')
    logger.info('spider dev data loaded.')

    pool = mp.Pool(os.cpu_count())

    grammar_file = 'bench/spider/grammar/spider_schema_grammar.json'
    print(f'using schema-specific grammar file from: {grammar_file}')
    with open(grammar_file, 'r') as f:
        all_grammars = json.load(f)

    # re-order the list of the grammars so that they're in the order they appear
    # in the dataset. Use dict over set because python guarantees insertion order in
    # dict, not in set.
    reordered_grammar_names = {}
    for dev_datum in spider_dev_data:
        if dev_datum.schema_name not in reordered_grammar_names:
            reordered_grammar_names[dev_datum.schema_name] = None

    async_guides = {}
    for schema_name in reordered_grammar_names:
        grammar = all_grammars[schema_name]
        grammar = reformat_grammar(grammar)
        async_guides[schema_name] = pool.apply_async(lark_guide, (grammar,))

    guides = {}

    for dev_datum in tqdm(spider_dev_data, desc='parse', smoothing=0.0):
        if dev_datum.schema_name not in guides:
            guide = async_guides[dev_datum.schema_name].get()
            guides[dev_datum.schema_name] = guide
            logger.info(f'parser for schema {dev_datum.schema_name} is ready.')
        else:
            guide = guides[dev_datum.schema_name]

        assert guide.p_next(dev_datum.query)['▪'] == 1

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
