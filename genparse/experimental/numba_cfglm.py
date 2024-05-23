"""
Fast computation of the posterior distrubtion over the next word in a WCFG language model.
"""
import numba
import numpy as np
from numba import jit
from arsenal import colors
from arsenal.maths import sample_dict
from collections import defaultdict

from genparse.cfg import _gen_nt, CFG, add_EOS, EOS
from genparse.semiring import Float


class CFGLM:

    def __init__(self, cfg, renumber=True):
        if EOS not in cfg.V: cfg = add_EOS(cfg)

        # cache columns of the chart indexed by prefix
        self._chart = {}

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

        self.pfg_array_grammar_N, self.pfg_array_grammar_T, self.pfg_array_N, self.pfg_array_V = array_grammar(self.pfg)

    # TODO: should probably be PCFGLM class, which is tied to semifields, rather
    # than CFGLM, which is meant to semiring-friendly.
    @classmethod
    def from_string(cls, x, semiring=Float, **kwargs):
        return cls(locally_normalize(CFG.from_string(x, Float), **kwargs))

    def chart(self, prefix):
        c = self._chart.get(prefix)
        if c is None:
            c = self._compute_chart(prefix)
            self._chart[prefix] = c
        return c

    def _compute_chart(self, prefix):
        if len(prefix) == 0:
            # TODO: double check this!
            tmp = [defaultdict(self.pfg.R.chart)]
            tmp[0][0][self.pfg.S] = self.pfg('')
            return tmp
        else:
            chart = self.chart(prefix[:-1])
            last_chart = extend_chart(self.pfg, chart, prefix)
            return chart + [last_chart]    # TODO: avoid list addition here as it is not constant time!

    def numba_chart(self, prefix):
        if len(prefix) == 0:
            return []
        else:
            chart = self.numba_chart(prefix[:-1])
            N = self.pfg_array_grammar_N
            T = self.pfg_array_grammar_T
            last_chart = extend_numba_chart(N, T, chart, prefix)
            return chart + [last_chart]    # TODO: avoid list addition here as it is not constant time!

    def p_next(self, prefix, NUMBA=True):

        if NUMBA: # in this case the fast computation is executed
            assert self.pfg.R == Float
            chart = self.numba_chart(prefix)

            start_index = self.pfg_array_N[self.pfg.S]
            N = self.pfg_array_grammar_N
            T = self.pfg_array_grammar_T

            print(chart,prefix)
            array_weights = numba_next_token_weights(N, T, chart, prefix, start_index)
            dict_weights = Float.chart()
            for a in self.pfg.V : # [index, weight] ==> [terminal, weight]
                dict_weights[a] = array_weights[self.pfg.ordered_V[a]]
            return dict_weights
        else:
            chart = self.chart(prefix)

        return next_token_weights(self.pfg, chart, prefix)

    # TODO: Use the cached charts for the prefix-transformed grammar to compute
    # the total probability of the string `x`.
    def __call__(self, x):
        assert x[-1] == EOS
        return self.cfg(x)

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


def _ordered_N(self):
    N_list = list(self.N)
    N_dict = {}
    for i in range(len(N_list)):
        N_dict[N_list[i]] = i
    return N_dict


def _ordered_V(self):
    V_list = list(self.V)
    V_dict = {}
    for i in range(len(V_list)):
        V_dict[V_list[i]] = i
    return V_dict


def array_grammar(self):
    """Returns two np arrays encoding the rules of the grammar
    (except the nullaries ). works only in cnf"""

    assert self.in_cnf()
    (nullary, terminal, binary) = self._cnf

    ordered_N = _ordered_N(self)
    ordered_V = _ordered_V(self)

    N = len(self.N)
    V = len(self.V)

    R = np.zeros([N,N,N])
    T = np.zeros([N,V])

    for r in binary:
        i = ordered_N[r.head]
        j = ordered_N[r.body[0]]
        k = ordered_N[r.body[1]]
        R[i,j,k] += r.w

    for rules in terminal.values():
        for r in rules:
            i = ordered_N[r.head]
            k = ordered_V[r.body[0]]
            T[i,k] += r.w

    return R, T, ordered_N, ordered_V


@jit
def extend_numba_chart(N, T, chart, prefix):
    """
    An O(N²) time algorithm to extend to the "chart" with the last token
    appearing at the end of "prefix"; returns a new chart column.
    """
    if len(chart) == 0:
        return

    k = len(prefix)

    # Nullary
    t = T.shape[1]
    n = N.shape[0]
    new = np.zeros([k, n])

    # Preterminal
    for a in range(0,t):
        for X in range(0,n):
            new[k-1,X] += T[X,a]

    # Binary rules
    for span in range(1, k+1):
        i = k - span
        for j in range(i + 1, k):
            for X,Y,Z in range(0,n):
                new[i,X] += chart[j][i,Y] * new[j,Z] * N[X,Y,Z]

    return new

@jit
def numba_next_token_weights(N, T, chart, prefix, start_index):
    """
    An O(N²) time algorithm to the total weight of a each next-token
    extension of `prefix`.
    """
    k = len(prefix) + 1
    t = T.shape[1]
    n = N.shape[0]

    if len(chart) != 0 :
        # the code below is just backprop / outside algorithm
        α = np.zeros([k,n])
        α[0][start_index] += 1

        # Binary rules
        for span in reversed(range(1, k + 1)):
            i = k - span
            for j in range(i + 1, k):
                    for X,Y,Z in range(0,n):
                        α[j,Z] += N[X,Y,Z] * chart[j][i,Y] * α[i, X]

    q = np.zeros(t)
    for X in range(0,n):
        for a in range(0,t):
            q[a] += T[X,a] * α[k-1,X]

    return q



#__________________________________
# TESTS


import genparse
import genparse.examples
#from genparse import CFG, Float

#from arsenal import colors
#from arsenal.maths import sample_dict
#from collections import defaultdict


def fast_posterior(cfg, prefix):
    return CFGLM(cfg).p_next(tuple(prefix))


def next_token_weights_slow(cfg, prefix):
    """
    Compute the posterior over the next word in O(V N³) time.
    """
    Z = cfg.prefix_weight(prefix)
    p = cfg.R.chart()
    for v in sorted(cfg.V):
        p[v] = cfg.prefix_weight([*prefix, v])
        if p[v] == cfg.R.zero: del p[v]
    return Z, p


#def next_token_weights_less_slow(cfg, prefix):
#    """
#    Compute the posterior over the next word in O(V N²) time.
#    """
#    N = len(prefix)
#    P = cfg.prefix_grammar.cnf
#    chart = P._parse_chart(prefix)
#    Z = chart[0, P.S, N]
#    p = {}
#    for v in sorted(cfg.V):
#        p[v] = extend_chart(P, chart, [*prefix, v])[0, P.S, N+1]
#        if p[v] == cfg.R.zero: del p[v]
#    return Z, p


def test_new_abcdx():

    cfg = add_EOS(CFG.from_string("""

    1: S -> a b c d
    1: S -> a b c x
    1: S -> a b x x
    1: S -> a x x x
    1: S -> x x x x

    """, Float))

    for prefix in ['', 'a', 'ab', 'abc', 'abcd', 'acbde']:
        print()
        print(colors.light.blue % prefix)
        want = next_token_weights_slow(cfg, prefix)[1]
        print(want)
        have = fast_posterior(cfg, prefix)
        print(have)
        err = have.metric(want)
        print(colors.mark(err <= 1e-5))
        assert err <= 1e-5, err


def test_new_palindrome():

    cfg = add_EOS(genparse.examples.palindrome_ab)

    for prefix in ['', 'a', 'ab']:
        print()
        print(colors.light.blue % prefix)
        want = next_token_weights_slow(cfg, prefix)[1]
        print(want)
        have = fast_posterior(cfg, prefix)
        print(have)
        err = have.metric(want)
        print(colors.mark(err <= 1e-5))
        assert err <= 1e-5


def test_new_papa():

    cfg = add_EOS(genparse.examples.papa)

    for prefix in [
            [],
            ['papa'],
            ['papa', 'ate'],
            ['papa', 'ate', 'the'],
            ['papa', 'ate', 'the', 'caviar'],
    ]:
        print()
        print(colors.light.blue % prefix)
        want = next_token_weights_slow(cfg, prefix)[1]
        print(want)
        have = fast_posterior(cfg, prefix)
        print(have)
        print(colors.mark(have.metric(want) <= 1e-5))
        assert have.metric(want) <= 1e-5


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
