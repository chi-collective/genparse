from arsenal import colors, iterview
from itertools import product

from genparse import CFGLM, EOS
from genparse.segmentation import segmentation_pfst, run_segmentation_test


def test_basics():

    alphabet = set('abc')

    test_strings = [''.join(x) for t in range(1, 7) for x in product(alphabet, repeat=t)]
    if 0:
        test_strings = [
            'aa',
            'acab',
            'abc',
            'abcabc',
            'abcab',
        ]

    contexts = {'a', 'b', 'c', 'abc'}

    print(colors.yellow % 'not prefix closed (flat)')
    C = segmentation_pfst(contexts, alphabet, canonical=False)
    for x in iterview(test_strings, transient=True):
        run_segmentation_test(C, x, contexts)

    print(colors.yellow % 'not prefix closed (canonical)')
    C = segmentation_pfst(contexts, alphabet, canonical=True)
    for x in iterview(test_strings, transient=True):
        run_segmentation_test(C, x, contexts)

    contexts = {'a', 'b', 'c', 'ab', 'abc'}

    print(colors.yellow % 'test simple (flat)')
    C = segmentation_pfst(contexts, alphabet, canonical=False)
    for x in iterview(test_strings, transient=True):
        run_segmentation_test(C, x, contexts)

    print(colors.yellow % 'test simple (canonical)')
    C = segmentation_pfst(contexts, alphabet, canonical=True)
    for x in iterview(test_strings, transient=True):
        run_segmentation_test(C, x, contexts)


def test_distortion():
    c = CFGLM.from_string("""

    1: S -> a
    1: S -> a a
    2: S -> a a a

    """)

    alphabet = {'a'}
    tokens = {'a', 'aa', 'aaa', EOS}
    f = lambda y: ''.join(y).strip(EOS)

    print(colors.yellow % 'non-canonical')
    T = segmentation_pfst(tokens, alphabet, canonical=False)

    have = (c.cfg @ T).language(100)
    want = c.cfg.language(100)

    have.project(f).assert_equal(want.project(f), verbose=True)

    print(colors.yellow % 'canonical')
    T = segmentation_pfst(tokens, alphabet, canonical=True)

    have = (c.cfg @ T).language(100)
    want = c.cfg.language(100)

    have.project(f).assert_equal(want.project(f), verbose=True)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
