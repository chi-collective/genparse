# -*- coding: utf-8 -*-

"""
Example usage: pyprof-callgraph scripts/benchmark_inference.py --particles 10
"""

import os
import time
import json
import psutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp

from bench.cache import PersistentGuideCache
from bench.spider.evaluator import Evaluator
from utils import (
    load_spider_data,
    load_prompt_formatter,
    reformat_grammar,
    HF_ACCESS_TOKEN,
    mbr_eval,
    posterior_weighted_eval,
    viterbi_eval,
)

from genparse.util import set_seed
from genparse.experimental.batch_inference import (
    BatchVLLM,
    BatchStepModel,
    smc,
    importance_sampling,
)
from genparse.experimental.batch_inference.proposal import CharacterBatchProposal
from genparse.experimental.batch_inference.steer import ParticleApproximation, Particle

os.environ['TOKENIZERS_PARALLELISM'] = '(true | false)'

logger = logging.getLogger(__name__)


def int_or_none(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Invalid integer value: {value}')


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct'
    )
    parser.add_argument('--particles', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument(
        '--proposal',
        choices=['character', 'token'],
        default='character',
        help='Specify which proposal distribution to use in SMC inference.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-processes', type=int_or_none, default=None)
    parser.add_argument(
        '--K',
        type=int_or_none,
        default=0,
        help='parameter for token proposal distribution',
    )

    return parser


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )

    parser = get_argparser()
    args = parser.parse_args()

    raw_spider_dir = Path('../spider/data/spider')
    spider_dev_data = load_spider_data(raw_spider_dir, split='dev')
    prompt_formatter = load_prompt_formatter(raw_spider_dir)

    with open('../../benchmark/grammars/spider_schema_grammar.json', 'r') as f:
        all_grammars = json.load(f)

    guide_cache = PersistentGuideCache('guide_cache.pkl')

    os.environ['HF_TOKEN'] = HF_ACCESS_TOKEN
    batch_llm = BatchVLLM.from_name(args.model_name)
    tokenizer = batch_llm.get_tokenizer()

    dev_datum = spider_dev_data[451]

    messages = prompt_formatter.format_openai(dev_datum)

    grammar = reformat_grammar(all_grammars[dev_datum.schema_name])

    guide = guide_cache.get(grammar)

    batch_proposal = CharacterBatchProposal(llm=batch_llm.llm, guide=guide)

    step_model = BatchStepModel(
        batch_proposal=batch_proposal,
        batch_llm=batch_llm,
        max_tokens=args.max_tokens,
    )

    prompt = tokenizer.decode(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    )

    step_model.set_prompt(prompt)

    smc(step_model, n_particles=args.particles, verbosity=1)


if __name__ == '__main__':
    main()
