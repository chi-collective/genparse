import psutil
import os

import argparse
import logging
from genparse.lark_interface import LarkStuff

from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_guide(grammar_name, use_fast=False):
    cfg = LarkStuff(open(grammar_name).read()).char_cfg()
    if use_fast:
        print('Using earley_fast')
        from genparse.experimental.earley_fast import BoolCFGLM
    else:
        from genparse.cfglm import BoolCFGLM
    return BoolCFGLM(cfg)


def load_examples(example_path):
    return [z[0:-1] for z in open(example_path, 'r').readlines()]


def main(grammar_path, example_path, out_path):
    logging.basicConfig(filename=out_path, level=logging.INFO)

    logger.info('Loading guide and examples')

    logger.info(f'guide : {grammar_path}')
    logger.info(f'examples : {example_path}')

    guide = load_guide(grammar_path, args.use_fast)
    exmps = load_examples(example_path)

    for ex_id, entry in enumerate(tqdm(exmps)):
        if not guide.p_next(entry):
            logger.info(f'FAILED {ex_id} : {entry}')
            for i in range(1, len(entry)):
                if not guide.p_next(f'{entry[0:i]}'):
                    logger.info(f'\t{entry[0:i]}')
                    break
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    ram_usage = memory_info.rss
    ram_usage_mb = ram_usage / (1024**2)
    print(f'RAM usage: {ram_usage_mb:.2f} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test character-level grammar coverage on a set of examples.'
    )
    parser.add_argument(
        '--grammar_path',
        type=str,
        default='../benchmark/grammars/sql_case_insensitive.lark',
    )
    parser.add_argument(
        '--example_path',
        type=str,
        help='txt file with one example per line',
        default='../benchmark/datasets/spider_dev_set.txt',
    )
    parser.add_argument('--out_path', type=str, help='output file')
    parser.add_argument('--use-fast', action='store_true')

    args = parser.parse_args()
    main(args.grammar_path, args.example_path, args.out_path)
