from genparse.segmentation import segmentation_pfst, run_segmentation_test
from itertools import product
from arsenal import colors, iterview


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


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
