from itertools import product

from arsenal import colors, iterview

from genparse import CFGLM, EOS
from genparse.segmentation import (longest_suffix_in, max_munch,
                                   run_segmentation_test, segmentation_pfst)


def test_basic_abc_noncanonical():

    alphabet = set("abc")
    test_strings = [
        "".join(x) for t in range(1, 7) for x in product(alphabet, repeat=t)
    ]
    contexts = {"a", "b", "c", "ab", "abc"}

    C = segmentation_pfst(contexts, alphabet, canonical=False)
    for x in iterview(test_strings, transient=True):
        run_segmentation_test(C, x, contexts)


def test_basic_abc_canonical():

    alphabet = set("abc")
    test_strings = [
        "".join(x) for t in range(1, 7) for x in product(alphabet, repeat=t)
    ]
    contexts = {"a", "b", "c", "ab", "abc"}

    C = segmentation_pfst(contexts, alphabet, canonical=True)
    for x in iterview(test_strings, transient=True):
        run_segmentation_test(C, x, contexts, canonical=True)


def test_not_prefix_closed_noncanonical():

    alphabet = set("abc")
    test_strings = [
        "".join(x) for t in range(1, 7) for x in product(alphabet, repeat=t)
    ]
    contexts = {"a", "b", "c", "abc"}

    C = segmentation_pfst(contexts, alphabet, canonical=False)
    for x in iterview(test_strings, transient=True):
        run_segmentation_test(C, x, contexts)


def test_abc_prefix_closed_canonical():

    alphabet = set("abc")
    test_strings = [
        "".join(x) for t in range(1, 7) for x in product(alphabet, repeat=t)
    ]
    contexts = {"a", "b", "c", "abc"}

    C = segmentation_pfst(contexts, alphabet, canonical=True)
    for x in iterview(test_strings, transient=True):
        run_segmentation_test(C, x, contexts, canonical=True)


def test_aaa_canonical():

    alphabet = set("a")
    test_strings = [
        "".join(x) for t in range(1, 7) for x in product(alphabet, repeat=t)
    ]
    contexts = {"a", "aa", "aaa"}

    C = segmentation_pfst(contexts, alphabet, canonical=True)
    for x in iterview(test_strings, transient=True):
        run_segmentation_test(C, x, contexts, canonical=True)


def test_aaa_noncanonical():

    alphabet = set("a")
    test_strings = [
        "".join(x) for t in range(1, 7) for x in product(alphabet, repeat=t)
    ]
    contexts = {"a", "aa", "aaa"}

    C = segmentation_pfst(contexts, alphabet, canonical=False)
    for x in iterview(test_strings, transient=True):
        run_segmentation_test(C, x, contexts)


def test_util():

    tokens = ["abc", "ab", "a", "b", "c"]
    t = max_munch(tokens)
    have = t("abcabc")
    want = ("abc", "abc")
    assert have == want, [have, want]

    tokens = ["aaa", "aa", "a"]
    t = max_munch(tokens)
    assert t("aaaaaa") == ("aaa", "aaa")
    assert t("aaaaa") == ("aaa", "aa")

    have = t("aaaa")
    want = ("aaa", "a")
    assert have == want, [have, want]

    assert t("aaa" "aaa" "aa") == ("aaa", "aaa", "aa")

    assert longest_suffix_in(["e", "de"])("abcde") == "de"
    assert longest_suffix_in([""])("abcde") == ""


def test_distortion():
    c = CFGLM.from_string(
        """

    1: S -> a
    1: S -> a a
    2: S -> a a a

    """
    )

    alphabet = {"a"}
    tokens = {"a", "aa", "aaa", EOS}
    f = lambda y: "".join(y).strip(EOS)

    print(colors.yellow % "non-canonical")
    T = segmentation_pfst(tokens, alphabet, canonical=False)

    have = (c.cfg @ T).language(100)
    want = c.cfg.language(100)

    have.project(f).assert_equal(want.project(f), verbose=True)

    print(colors.yellow % "canonical")
    T = segmentation_pfst(tokens, alphabet, canonical=True)

    have = (c.cfg @ T).language(100)
    want = c.cfg.language(100)

    have.project(f).assert_equal(want.project(f), verbose=True)


# def test_bpe():
#    import numpy as np
#    from genparse.util import hf_tokenizer
#
#    H = hf_tokenizer()
#
#    _, B = zip(*H.pairs)
#    B = list(B)
#
#    np.random.shuffle(B)
#    B = B[:500]
#
#    A = {c for b in B for c in b}
#
#    B = set(B) | A
#
#    T = segmentation_pfst(B, A, canonical=True).trim
#
#    print(T.dim, 'states')


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
