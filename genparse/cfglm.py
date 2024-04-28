"""
Fast computation of the posterior distrubtion over the next word in a WCFG language model.
"""

import numpy as np
from functools import lru_cache
from collections import Counter, defaultdict
from arsenal.maths import sample_dict

from .cfg import CFG, _gen_nt
from . import Chart
from .semiring import Float


def locally_normalize(self, **kwargs):
    """
    Locally normalizes the grammar: return a transformed grammar such that

    (1) the total weight of each block of rules with a common head sum to one;

    (2) each derivation in the transformed grammar is proportional to the original grammar
        (i.e., it is has the same score modulo a multiplicative normalization constant)

    """
    new = self.spawn()
    Z = self.agenda(**kwargs)
    for r in self:
        if Z[r.head] == 0: continue
        new.add(r.w * Z.product(r.body) / Z[r.head], r.head, *r.body)
    return new


class CFGLM:

    def __init__(self, cfg):
        self.cfg = cfg.renumber()
        self.pfg = cfg.cnf.prefix_grammar.cnf.renumber().cnf

        self.pfg.r_y_xz = r_y_xz = defaultdict(list)
        for r in self.pfg._cnf[2]:  # binary rules
            r_y_xz[r.body[0]].append(r)

    @lru_cache(None)
    def chart(self, prefix):
        if len(prefix) == 0:
            # TODO: double check this!
            tmp = [defaultdict(self.pfg.R.chart)]
            tmp[0][0][self.pfg.S] = self.pfg('')
            return tmp
        else:
            chart = self.chart(prefix[:-1])
            last_chart = extend_chart(self.pfg, chart, prefix)
            return chart + [last_chart]    # TODO: avoid list addition here as it is not constant time!

    @lru_cache(None)
    def p_next(self, prefix):
        chart = self.chart(prefix)
        return next_token_weights(self.pfg, chart, prefix)

    def sample(self, draw=sample_dict, verbose=False):
        ys = []
        while True:
            p = self.p_next(tuple(ys))
            if verbose: print(ys)
            y = draw(p)
            if y == EOS: return ys
            ys.append(y)


def next_token_weights(cfg, chart, prefix):
    """
    An O(N²) time algorithm to extend to the `chart` with the last token
    appearing at the end of `s`; returns a new chart.
    """
    k = len(prefix) + 1

    (nullary, terminal, binary) = cfg._cnf

    # the code below is just backprop / outside algorithm
    α = defaultdict(lambda: cfg.R.chart())
    α[0][cfg.S] += cfg.R.one

    # Binary rules
    for span in reversed(range(1, k + 1)):
        i = k - span
        for j in range(i + 1, k):   # TODO: use the same left-child index
            chart_j = chart[j]
            for r in binary:
                X, [Y, Z] = r.head, r.body
                α[j][Z] += r.w * chart_j[i][Y] * α[i][X]

    # Preterminal
    q = cfg.R.chart()
    for w in cfg.V:
        for r in terminal[w]:
            q[w] += r.w * α[k-1][r.head]

    return q


def extend_chart(cfg, chart, s):
    """
    An O(N²) time algorithm to extend to the `chart` with the last token
    appearing at the end of `s`; returns a new chart.
    """
    k = len(s)

    (nullary, terminal, binary) = cfg._cnf
    r_y_xz = cfg.r_y_xz

    new = defaultdict(lambda: cfg.R.chart())

    # Nullary
    new[k][cfg.S] += nullary

    # Preterminal
    for r in terminal[s[k-1]]:
        new[k-1][r.head] += r.w

    # Binary rules
    for span in range(1, k+1):
        i = k - span
        new_i = new[i]
        for j in range(i + 1, k):
            chart_ij = chart[j][i]
            new_j = new[j]
            for Y, Y_score in chart_ij.items():
                for r in r_y_xz[Y]:
                    X, [Y, Z] = r.head, r.body
                    new_i[X] += r.w * Y_score * new_j[Z]

    return new


# TODO: Make the token-id sequences available as well as the character
# sequences.  Using the character sequences is useful the CFGLM caching, so we
# should not dispense with it!
class CharAlignedCFGLM:
    """
    This class implements a simple strategy for "aligning" a character-level
    CFG language model to a vocabulary of character chunks, such as those
    used in the common byte-pair encoding (BPE) schemes of large language models.
    """

    def __init__(self, lm, words, eos):

        # TODO: Correctly handle the possibility that the word model and lm may
        # have different EOS symbols. To accomodate this, we just we need
        # something that converts them that they don't have to be equal strings.
        # This will just amount to some special cases.
        assert eos in words

        self.lm = lm
        self.words = words
        self.eos = eos
        self._end = object()
        self.trie = self.make_trie(words)

    def make_trie(self, words):
        root = dict()
        for word in words:
            curr = root
            for letter in word:
                curr = curr.setdefault(letter, {})
            curr[self._end] = self._end
        return root

    def p_next(self, context):
        t = len(context)
        return Float.chart(
            # strip the common length-t prefix
            (k[t:], v) for k,v in self.traverse_trie(context, self.trie, 1)
        ).normalize()

    def traverse_trie(self, context, node, P):
        p = self.lm.p_next(context)
        for x in node.keys():
            if x == self._end:
                yield (context, P)
                continue
            P_x = P * p[x]
            if P_x == 0: continue
            yield from self.traverse_trie(context + x, node[x], P_x)

    def traverse_naive(self, context, node, P):
        for x in self.words:
            p = self.lm.pfg(context + x)
            P_x = P * p
            if P_x == 0: continue
            yield (context + x, P_x)

    def sample(self, draw=sample_dict, verbose=False):
        context = ''
        while True:
            if verbose: print(repr(context))
            p = self.p_next(context)
            y = draw(p)
            if y == self.eos: break
            context += y
            # TODO: this is an ugly hack the arises from sloppy handling of EOS.
            # To handle this cleanly we just need to align the EOS in the LM and
            # the EOS in words.
            if context.endswith('</s>'): break
        if verbose: print(repr(context))
        return context


#EOS = '$EOS'
#EOS = '🛑'
EOS = '▪'


# spacer is a kind of end-of-token marker
#SPACER = '$SPACER'
SPACER = '#'
#EOT = '#'


# TODO: better to use concatenation of grammars!
def add_EOS(cfg, EOS=EOS):
    "Append the EOS symbol to the language generated by `cfg`."
    S = _gen_nt('<START>')
    new = cfg.spawn(S=S)
    assert EOS not in cfg.V
    new.V.add(EOS)
    new.add(cfg.R.one, S, cfg.S, EOS)
    for r in cfg:
        new.add(r.w, r.head, *r.body)
    return new


def explode_terminal(y):
    assert y != SPACER
    if y == EOS:      # don't explode EOS!
        return [y]
    else:
        return list(y + SPACER)


def explode(cfg):
    """
    Explode the terminals into characters - this is a special case of composition
    with a transducer.
    """
    new = cfg.spawn(V = set())
    for r in cfg:
        ys = []
        for y in r.body:
            if cfg.is_terminal(y):
                yy = explode_terminal(y)
                ys.extend(yy)
                new.V |= set(yy)
            else:
                ys.append(y)
        new.add(r.w, r.head, *ys)
    return new
