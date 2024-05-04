"""
Fast computation of the posterior distrubtion over the next word in a WCFG language model.
"""

from arsenal import colors
from arsenal.maths import sample_dict
from collections import defaultdict
from functools import lru_cache

from .cfg import _gen_nt, CFG
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

    def __init__(self, cfg, renumber=True):
        if EOS not in cfg.V: cfg = add_EOS(cfg)
        if renumber:
            self.cfg = cfg.renumber()
            self.pfg = cfg.cnf.prefix_grammar.cnf.renumber().cnf
        else:
            self.cfg = cfg
            self.pfg = cfg.cnf.prefix_grammar.cnf

        # TODO: this is is a quick hack; clean this up.
        self.pfg.r_y_xz = r_y_xz = defaultdict(list)
        for r in self.pfg._cnf[2]:  # binary rules
            r_y_xz[r.body[0]].append(r)

    # TODO: should probably be PCFGLM class, which is tied to semifields, rather
    # than CFGLM, which is meant to semiring-friendly.
    @classmethod
    def from_string(cls, x, semiring=Float, **kwargs):
        return cls(locally_normalize(CFG.from_string(x, Float), **kwargs))

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

    def sample(self, draw=sample_dict, prob=False, verbose=False):
        ys = ()
        P = 1.0
        while True:
            p = self.p_next(ys).normalize()
            if verbose: print(ys)
            y = draw(p)
            P *= p[y]
            if y == EOS:
                return (ys, P) if prob else ys
            ys = ys + (y,)


def next_token_weights(cfg, chart, prefix, alpha=False):
    """
    An O(N²) time algorithm to the total weight of a each next-token
    extension of `prefix`.
    """
    k = len(prefix) + 1

    (_, terminal, _) = cfg._cnf
    r_y_xz = cfg.r_y_xz

    # the code below is just backprop / outside algorithm
    α = defaultdict(cfg.R.chart)
    α[0][cfg.S] += cfg.R.one

    # Binary rules
    for span in reversed(range(1, k + 1)):
        i = k - span
        α_i = α[i]
        for j in range(i + 1, k):
            chart_ij = chart[j][i]

            α_j = α[j]
            for Y, y in chart_ij.items():
                for r in r_y_xz[Y]:
                    X = r.head
                    Z = r.body[1]
                    α_j[Z] += r.w * y * α_i[X]

    # Preterminal
    q = cfg.R.chart()
    tmp = α[k-1]
    for w in cfg.V:
        for r in terminal[w]:
            q[w] += r.w * tmp[r.head]

    if alpha:
        return q, α
    else:
        return q


def extend_chart(cfg, chart, prefix):
    """
    An O(N²) time algorithm to extend to the `chart` with the last token
    appearing at the end of `prefix`; returns a new chart column.
    """
    k = len(prefix)

    (nullary, terminal, _) = cfg._cnf
    r_y_xz = cfg.r_y_xz

    new = defaultdict(cfg.R.chart)

    # Nullary
    new[k][cfg.S] += nullary

    # Preterminal
    for r in terminal[prefix[k-1]]:
        new[k-1][r.head] += r.w

    # Binary rules
    for span in range(1, k+1):
        i = k - span
        new_i = new[i]
        for j in range(i + 1, k):
            chart_ij = chart[j][i]
            new_j = new[j]
            for Y, y in chart_ij.items():
                for r in r_y_xz[Y]:
                    X = r.head
                    Z = r.body[1]
                    z = new_j[Z]
                    x = r.w * y * z
                    new_i[X] += x

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
        return self.lm.cfg.R.chart(
            # strip the common length-t prefix
            (k[t:], v) for k,v in self.traverse_trie(context, self.trie, 1)
        ).normalize()

    def traverse_trie(self, context, node, P):
        p = self.lm.p_next(context)
        for x in node:
            if x == self._end:
                yield (context, P)
#                yield (context, self.lm.pfg(context))
                continue
            P_x = P * p[x]
            if P_x == 0: continue
            yield from self.traverse_trie(context + x, node[x], P_x)

    # TODO: test equivalence of `traverse_trie` and `traverse_naive`.
    def traverse_naive(self, context):
        for x in self.words:
            p_x = self.lm.pfg(context + x)  # prefix weight of context + x
            if p_x == 0: continue
            yield (context + x, p_x)

    def sample(self, draw=sample_dict, prob=False, chunked=False, verbose=False):
        context = ''
        chunks = []
        P = 1
        while True:
            if verbose: print(repr(context))
            p = self.p_next(context).normalize()
            y = draw(p)
            P *= p[y]
            if y == self.eos: break
            chunks.append(y)
            context += y
            # TODO: this is an ugly hack the arises from sloppy handling of EOS.
            # To handle this cleanly we just need to align the EOS in the LM and
            # the EOS in words.
            if context.endswith('</s>'): break
        if verbose: print(repr(context))
        value = context
        if chunked: value = tuple(chunks)
        if prob: value = (value, P)
        return value


#EOS = '$EOS'
#EOS = '🛑'
EOS = '▪'


# spacer is a kind of end-of-token marker
#SPACER = '$SPACER'
SPACER = '#'
#EOT = '#'


# TODO: better to use concatenation of grammars!
def add_EOS(cfg):
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


def pcfg_check(cfg):
    chart = cfg.agenda()
    if all((0 <= v <= 1.000001) for v in chart.values()):
        print(colors.mark(True), 'PCFG')
    else:
        print(colors.mark(False), 'PCFG', chart.__str__(style_value=lambda k, v: v if abs(1 - v) <= 1e-5 else (colors.light.red % v)))


def cfg_check_bounded(cfg, ub=1.000001, lb=0):
    chart = cfg.agenda()
    if all((lb <= v <= ub) for v in chart.values()):
        print(colors.mark(True), 'PCFG')
    else:
        print(colors.mark(False), 'PCFG', chart.__str__(style_value=lambda k, v: v if lb <= v <= ub else (colors.light.red % v)))
