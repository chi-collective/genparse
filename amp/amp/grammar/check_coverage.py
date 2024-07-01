import argparse

from lark import Lark
from tqdm import tqdm

from ambiguous_parsing.generation.generate_pairs import generate_all_pairs


def main(args):
    with open('cfgs/lark/lark.cfg') as f:
        grammar = Lark(f, start='sent')
    all_pairs = generate_all_pairs()
    for name, pairs in all_pairs.items():
        if args.n:
            _pairs = pairs[: args.n]
        try:
            for p in tqdm(_pairs, desc=name):
                grammar.parse(p['lf'])
        except Exception:
            print('Some pairs failed to parse for ', name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=False)
    args = parser.parse_args()
    main(args)
