import pylab as pl
import argparse
import logging
from time import time
from arsenal import iterview, timers, timeit, colors
from arsenal.iterextras import unique
from genparse.segmentation import prefixes

#from genparse.cfglm import EarleyBoolMaskCFGLM
from genparse.util import LarkStuff

from genparse.experimental import earley1
from genparse.experimental import earley2
from genparse.experimental import earley3
from genparse.experimental import earley4
from genparse.experimental import earley5
from genparse.experimental import earley6


def load_examples(example_path):
    return unique(map(str.strip, open(example_path, 'r')))   # XXX: why are there duplicates?

def main(grammar_path, example_path, out_path):

    with timeit('preprocessing'):
        cfg = LarkStuff(open(grammar_path).read()).char_cfg(0.9, ignore='[ ]?')
        guide4 = earley4.EarleyLM(cfg)
        guide6 = earley6.EarleyLM(cfg)

    T = timers()

    start = time()
    for i, example in iterview(list(enumerate(load_examples(example_path)))[:10]):
        print(example)

        guide4.clear_cache()
        guide6.clear_cache()

        for prefix in prefixes(example):

            with T['earley4'](n=len(prefix)):
                p = guide4.p_next(prefix)

            if not p: print(colors.light.red % f'FAILED {i}: {prefix}')

            with T['earley6'](n=len(prefix)):
                p = guide6.p_next(prefix)

            if not p: print(colors.light.red % f'FAILED {i}: {prefix}')

    print('total time:', time() - start, 'seconds')

    T.compare()

#    T.plot_feature('n')
#    pl.show()


if __name__ == '__main__':
    from path import Path
    import genparse
    root = Path(genparse.__file__).dirname() / '..'

    parser = argparse.ArgumentParser(
        description='Test character-level grammar coverage on a set of examples.'
    )
    parser.add_argument(
        '--grammar_path',
        type=str,
        default=root / 'benchmark/grammars/sql_case_insensitive.lark',
    )
    parser.add_argument(
        '--example_path',
        type=str,
        help='txt file with one example per line',
        default=root / 'benchmark/datasets/spider_dev_set.txt',
    )
    parser.add_argument('--out_path', type=str, help='output file')

    args = parser.parse_args()
    main(args.grammar_path, args.example_path, args.out_path)
