"""
Probabiliistic finite-state transducer (PFST) that implements a segmentation model,
such as byte-pair encoding (BPE).
"""
import numpy as np
from arsenal import colors
from collections import defaultdict, Counter
from genparse import Float, FST, EPSILON, EOS
from genparse.util import display_table, HTML


fmt = lambda x: ''.join(x) or 'ε'


def prefixes(z):
    """
    Return the prefixes of the sequence `z`

      >>> list(prefixes(''))
      ['']

      >>> list(prefixes('abc'))
      ['', 'a', 'ab', 'abc']

    """
    for p in range(len(z)+1):
        yield z[:p]


def suffixes(z):
    """
    Return the prefixes of the sequence `z`

      >>> list(suffixes(''))
      ['']

      >>> list(suffixes('abc'))
      ['', 'c', 'bc', 'abc']

    """
    for p in reversed(range(len(z)+1)):
        yield z[p:]


class max_munch:

    def __init__(self, tokens):
        self._end = object()
        self.root = self.make_trie(tokens)

    def __call__(self, x):
        if len(x) == 0:
            return ()
        else:
            t, ys = next(self.traverse(x, 0, self.root))
            return (ys,) + self(x[t:])

    def make_trie(self, words):
        root = dict()
        for word in words:
            curr = root
            for letter in word:
                curr = curr.setdefault(letter, {})
            curr[self._end] = self._end
        return root

    def traverse(self, query, t, node):
        """
        Enumerate (in order of longest to shortest) the strings in the trie matching
        prefixes of `query`.
        """
        if node == self._end: return
        if t < len(query):
            x = query[t]
            if x in node:
                yield from self.traverse(query, t + 1, node[x])
        if self._end in node:
            yield (t, query[:t])   # post order gives the longest match


class longest_suffix_in:
    def __init__(self, C):
        self.rtrie = max_munch([reversed(c) for c in C])

    def __call__(self, c):
        r = tuple(reversed(c))
        _, ys = next(self.rtrie.traverse(r, 0, self.rtrie.root))
        return ''.join(reversed(ys)) if isinstance(c, str) else tuple(reversed(ys))


def segmentation_counting_wfst(S):
    """
    Note: This was the first attempt at a segmentation WFST.  It does not define
    a conditional distrbution over strings, it only defines a (uniform) weight
    over segmentations.  When applied to a given string, it will count the
    number of segmentations that string has under the set of allowed segments
    `S`.
    """
    m = FST(Float)
    m.set_I((), 1)
    S = [(a, tuple(b)) for a, b in S]
    for i, x in S:
        for j in range(len(x)):
            m.set_arc(x[:j], (x[j], EPSILON), x[:j+1], 1)
        m.set_arc(x, (EPSILON, i), (), 1)
    m.set_F((), 1)
    return m


def bpe_wfst(S, renumber=True):
    m = FST(Float)
    m.set_I((), 1)
    for i, x in S:
        x = tuple(x)
        for j in range(len(x)):
            m.set_arc(x[:j], (EPSILON, x[j]), x[:j+1], 1)
        m.set_arc(x, (i, EPSILON), (), 1)
    m.set_F((), 1)
    return m.renumber if renumber else m


def char2bpe_wfst(S, renumber=True):
    m = FST(Float)
    m.set_I((), 1)
    for i, x in S:
        x = tuple(x)
        for j in range(len(x)):
            m.set_arc(x[:j], (x[j], EPSILON), x[:j+1], 1)
        m.set_arc(x, (EPSILON, i), (), 1)
    m.set_F((), 1)
    return m.renumber if renumber else m


# NOTE: Below is an older, less-efficient construction: It is less efficient because doesn't
# share common prefixes on the character side (the new construction has about ~4x fewer states
# on the gpt2 tokenizer)
# def bpe_wfst(S):
#     "Create a transducer relating strings of BPE token ids to their associated strings"
#     from genparse import Float, FST, EPSILON
#     m = FST(Float)
#     START = 0
#     STOP = 1
#     m.set_I(0, 1)
#     for i, x in S:
#         m.set_arc(START, (i, EPSILON), (i, 0), 1)
#         for j in range(len(x)):
#             m.set_arc((i,j), (EPSILON, x[j]), (i,j+1), 1)
#         m.set_arc((i,len(x)), (EPSILON, EPSILON), STOP, 1)
#     m.set_F(STOP, 1)
#     m.set_arc(STOP, (EPSILON, EPSILON), START, 1)
#     return m.renumber


def segmentation_pfst(contexts, alphabet, canonical, debug=False, trim=True):
    """Probabilistic FST segmentation model; returns a model that satsified the
    conditions in`run_segmentation_test` for all input strings `x` over the
    alphabet.

    Technical conditions: We require alphabet <= contexts in order to ensure
    that no strings over this alphabet will dead end (i.e., have no
    segmentations).

    [Note: another workaround for the lack of prefix closure is to generate UNK]

    If `canonical = True`, we will return a segmentation model that places
    probabilty one on a single canonical segmentation that corresponds to using
    the fewest segments, as they are greedily matched in the construction.

    """
    assert EOS not in alphabet
    assert set(alphabet) <= set(contexts)

    #contexts = prefix_closure(contexts)

    C = {tuple(c): ''.join(c) for c in contexts}

    muncher = max_munch(C)

    states = {p for c in C for p in prefixes(c[:-1])}  # states track (proper) prefixes of contexts

    longest_suffix = longest_suffix_in(states)

    def gensym():
        gensym.id += 1
        return f'${gensym.id}'
    gensym.id = -1

    m = FST(Float)
    m.set_I((), 1)
    for p in sorted(states):
        for y in sorted(alphabet):

            # The current context
            c = p+(y,)

            # We can transition to any suffix of `c` that is also a state.
            next_states = {longest_suffix(c)} if canonical else set(suffixes(c)) & states

            # Consider `abcde + f = abcdef` that can backoff to any suffix that is also a state.
            #
            # if we pick the suffix `ε`, then we emit `abcdef` (or some segmentation of it with `|`)
            #
            # if we pick the suffix `def`, then we emit `abc` (or some segmentation of it with `|`)

            N = len(next_states)
            for q in sorted(next_states):

                emit = c if len(q) == 0 else c[:-len(q)]

                assert p + (y,) == emit + q   # invariant: ensures no information is lost.

                # TODO: we can segment the residual string in other ways; make
                # that possible.  Same for the EOS case.

                chunks = muncher(emit)
                if len(chunks) > 1:
                    # Suppose mm = c_1 | c_2 | ... | c_K
                    #
                    # m.set_arc(p, (y, c_1), tmp1, 1/N)
                    # m.set_arc(tmp1, ('', c_2), tmp2, 1/N)
                    # ...
                    # m.set_arc(tmp_{K-1}, ('', c_K), q, 1/N)
                    #
                    # p --a:c_1-->tmp1--ε:c_2-->tmp2 -- ... --> tmp_{K-1}--ε:c_K--> q

                    curr = p
                    for k, c_k in enumerate(chunks):

                        tmp = gensym() if k < len(chunks) - 1 else q  # last chunk; use q instead of tmp

                        if k == 0:
                            m.set_arc(curr, (y, ''.join(c_k)), tmp, 1/N)
                        else:
                            m.set_arc(curr, ('', ''.join(c_k)), tmp, 1)

                        curr = tmp

                else:
                    m.set_arc(p, (y, ''.join(emit)), q, 1/N)


        chunks = muncher(p)
        if len(chunks) > 1:

            curr = p
            for k, c_k in enumerate(chunks):

                tmp = gensym() if k < len(chunks) - 1 else EOS  # last chunk; use q instead of tmp

                if k == 0:
                    m.set_arc(curr, (EOS, ''.join(c_k)), tmp, 1)
                else:
                    m.set_arc(curr, ('', ''.join(c_k)), tmp, 1)

                curr = tmp

        else:
            m.set_arc(p, (EOS, ''.join(p)), EOS, 1)

    m.set_F(EOS, 1)
    return m.trim if trim else m


def run_segmentation_test(T, x, contexts, rmeps=True, maxseglenth=100, verbose=0, canonical=False):
    """Testing utility, verifies the following properties of the transducer `T` when
    applied to a given string `x`:

    1) `x @ T` is a valid PFST, meaning it defines a distribution over segmentations.

    2) `x @ T` the strings in the support are valid segmentations (i.e.,
       concatenating them results in the original string `x`).

    3) TODO: Check that segmentations have their intended support! This is
       application dependent, we should consider the following:

       (a) Support on *all* valid segmentations

       (b) Support on a single segmentation (e.g., min-munch, max-munch)

    """

    tmp = T(x + EOS, None)
    if rmeps: tmp = tmp.epsremove.trim
    Z = tmp.total_weight()
    if verbose: print(colors.mark(np.allclose(Z, 1)), 'total weight:', Z)
    D = tmp.to_cfg().cnf.language(maxseglenth)
    if verbose > 1: display_table([[D, tmp]])
    ok = True
    for Y in D:
        X = ''.join(Y)
        if verbose: print(colors.mark(X == x), X, Y)
        assert all((y in contexts) for y in Y)
        ok &= X == x
    assert ok and np.allclose(Z, 1) and (not canonical or len(D) == 1), f"""
        {x = }, {Z = }, {canonical = }, {D = }
    """
