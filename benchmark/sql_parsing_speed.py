from path import Path
import genparse
import pylab as pl
import argparse
from time import time
from arsenal import iterview, timers, timeit, colors
from arsenal.iterextras import unique
from genparse.segmentation import prefixes

from genparse.lark_interface import LarkStuff

from genparse import EarleyLM
from genparse.cfglm import BoolCFGLM


def load_examples(example_path):
    return unique(
        map(str.strip, open(example_path, 'r'))
    )  # XXX: why are there duplicates?


def main():
    root = Path(genparse.__file__).dirname() / '..'

    parser = argparse.ArgumentParser(
        description='Test character-level grammar coverage on a set of examples.'
    )
    parser.add_argument(
        '--grammar',
        type=Path,
        default=root / 'benchmark/grammars/sql_case_insensitive.lark',
    )
    parser.add_argument(
        '--examples',
        type=Path,
        help='text file with one example per line',
        default=root / 'benchmark/datasets/spider_dev_set.txt',
    )
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    guide = {}
    with timeit('lark'):
        left = LarkStuff(open(args.grammar).read()).char_cfg(recursion='left')
        right = LarkStuff(open(args.grammar).read()).char_cfg(recursion='right')
    with timeit('preprocessing'):
        # guide['earley'] = EarleyLM(cfg)
        guide['bool-earley-left'] = BoolCFGLM(left)
        guide['bool-earley-right'] = BoolCFGLM(right)
        # guide['cfglm'] = CFGLM(cfg)

    T = timers()

    start = time()
    for i, example in iterview(list(enumerate(load_examples(args.examples)))[:20]):
        print(example)

        for name in guide:
            guide[name].clear_cache()

        for prefix in prefixes(example):
            for name in guide:
                with T[name](n=len(prefix)):
                    p = guide[name].p_next(prefix)

                if not p:
                    print(colors.light.red % f'FAILED {i}: {prefix}')

    print('total time:', time() - start, 'seconds')

    T.compare()

    if args.plot:
        T.plot_feature('n')
        pl.show()


if __name__ == '__main__':
    main()
