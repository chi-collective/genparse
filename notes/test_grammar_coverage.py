import sys
import getpass
import argparse
import logging

logger = logging.getLogger(__name__)

if getpass.getuser() == "benjamin.lebrun": 
    sys.path.append("/home/mila/b/benjamin.lebrun/genparse")

from genparse.util import LarkStuff
from genparse.cfglm import EarleyBoolMaskCFGLM

def load_guide(grammar_name):
    cfg = LarkStuff(open(grammar_name).read()).char_cfg(.99, ignore='[ ]?')
    return EarleyBoolMaskCFGLM(cfg)

def load_examples(example_path):
    return [z[0:-1] for z in open(example_path, "r").readlines()]

def main(grammar_path, example_path, out_path):
    logging.basicConfig(filename=out_path, level=logging.INFO)
    
    logger.info('Loading guide and examples')

    logger.info(f'guide : {grammar_path}')
    logger.info(f'examples : {example_path}')
    
    guide = load_guide(grammar_path)
    exmps = load_examples(example_path)

    for ex_id, entry in enumerate(exmps):
        if not guide.p_next(entry):
            logger.info(f"FAILED {ex_id} : {entry}")
            for i in range(1, len(entry)):
                if not guide.p_next(f'{entry[0:i]}'):
                    logger.info(f'\t{entry[0:i]}')
                    break

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test character-level grammar coverage on a set of examples.")
    parser.add_argument("--grammar_path", type=str, default='../benchmark/grammars/sql_case_insensitive.lark')
    parser.add_argument("--example_path", type=str, help="txt file with one example per line", default="../benchmark/datasets/spider_dev_set.txt")
    parser.add_argument("--out_path", type=str, help="output file")

    args = parser.parse_args()
    main(args.grammar_path, args.example_path, args.out_path)
    
