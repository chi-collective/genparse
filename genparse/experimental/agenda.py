"""
Fast computation of the posterior distrubtion over the next word in a WCFG language model.
"""

from arsenal import colors
from arsenal.maths import sample_dict
from collections import defaultdict

from genparse.cfg import _gen_nt, CFG
from genparse.semiring import Float
from genparse.cfglm import add_EOS, EOS


import genparse.cfglm


# TODO: This method can still be improved in many ways...
class AgendaCFGLM(genparse.cfglm.CFGLM):

    def __init__(self, cfg, renumber=True):
        super().__init__(cfg, renumber=renumber)
        self.build_indexes()

    def build_indexes(self):
        self.pfg.r_y_xz = r_y_xz = defaultdict(list)
        for r in self.pfg._cnf[2]:  # binary rules
            r_y_xz[r.body[0]].append(r)

        self.pfg.r_z_xy = r_z_xy = defaultdict(list)
        for r in self.pfg._cnf[2]:  # binary rules
            r_z_xy[r.body[1]].append(r)

        self.pfg.r_yz_x = r_yz_x = defaultdict(list)
        for r in self.pfg._cnf[2]:  # binary rules
            r_yz_x[r.body].append(r)

    def _compute_chart(self, prefix):
        if len(prefix) == 0:
            # TODO: double check this!
            tmp = [defaultdict(self.pfg.R.chart)]
            tmp[0][0][self.pfg.S] = self.pfg('')
            return tmp
        else:
            chart = self.chart(prefix[:-1])
            last_chart = extend_chart_sparse(self.pfg, chart, prefix)
            return chart + [last_chart]    # TODO: avoid list addition here as it is not constant time!

    # TODO: also try to speedup p_next with the agenda-based implementation
#    def p_next(self, prefix):
#        chart = self.chart(prefix)
#        return genparse.cfglm.next_token_weights(self.pfg, chart, prefix)


#import numpy as np
#from arsenal import timers
#TIMER = timers()
#def extend_chart(cfg, chart, prefix):
##    with TIMER['sparse']:
##        sparse = extend_chart_sparse(cfg, chart, prefix)
#
#    with TIMER['dense']:
#        dense = extend_chart_dense(cfg, chart, prefix)
#
#    if cfg.R == Float:
#        score = lambda x: x
#        Score = lambda x: x
#    else:
#        score = lambda x: x.score
#        Score = lambda x: cfg.R(x)
#
#    from genparse import _cfglm
#    cfg = cfg.renumber()
#    nullary = score(cfg._cnf[0])
#    terminal = {Y: [(score(r.w), r.head) for r in cfg._cnf[1][Y]] for Y in cfg._cnf[1]}
#    binary = [_cfglm.Rule(score(r.w), r.head, r.body[0], r.body[1]) for r in cfg._cnf[2]]
#
#    _chart = []
#    for k, c_k in enumerate(chart):
#        C_k = np.zeros((k+1, len(cfg.N)))
#        for i in c_k:
#            for X, v in c_k[i].items():
#                C_k[i, X] += score(v)
#        _chart.append(C_k)
#
#    with TIMER['cython']:
#        _cython = _cfglm.extend_chart(cfg.S, len(cfg.N), nullary, terminal, binary, _chart, tuple(prefix))
#
#    cython = defaultdict(cfg.R.chart)
#    for i in range(k+1):
#        for X in range(len(cfg.N)):
#            cython[i][X] = Score(C_k[i, X])   # only if its Real
#
#    TIMER.compare()
#    #print(sparse)
#    #print(dense)
#    #from genparse import Chart
#    #_sparse = cfg.R.chart(flatten_chart(sparse, Chart))
#    #_dense = cfg.R.chart(flatten_chart(dense, Chart))
#    #_sparse.assert_equal(_dense, tol=1e-8)
#
#    return dense
##    return cython


#def flatten_chart(x, Chart):
#    if isinstance(x, Chart):
#        return x
#    elif isinstance(x, list):
#        return flatten_chart_chart(dict(enumerate(x)))
#    else:
#        tmp = dict()
#        for i, y in x.items():
#            for j, v in flatten_chart(y, Chart).items():
#                tmp[i, j] = v
#        return tmp

#
# TODO: improve the column data strucure so that all(len(chart[j][i]) >
# 0 for i in chart[j]).  The problem is the usual one with defaultdict.
#
# TODO: improve repropagation by using better prioritization
#
def extend_chart_sparse(cfg, chart, prefix):
    """
    An O(N²) time algorithm to extend to the `chart` with the last token
    appearing at the end of `prefix`; returns a new chart column.
    """
    k = len(prefix)

    (nullary, terminal, binary) = cfg._cnf
    #r_y_xz = cfg.r_y_xz
    #r_z_xy = cfg.r_z_xy
    r_yz_x = cfg.r_yz_x

    new = defaultdict(cfg.R.chart)

#    agenda = []
#    _agenda = {}
    _agenda = defaultdict(set)

#    from heapq import heappush, heappop

    def push(i, X):
#        _agenda_i = _agenda.get(i)
#        if _agenda_i is None:
#            _agenda[i] = _agenda_i = set()
#        if X in _agenda_i: return
#        _agenda_i.add(X)
#        heappush(agenda, -i)
        #agenda.add(i)
        _agenda[i].add(X)


    zero = cfg.R.zero

    # Nullary
    if nullary != zero:
        new[k][cfg.S] += nullary

    # Preterminal
    for r in terminal[prefix[k-1]]:
        new[k-1][r.head] += r.w
        push(k-1, r.head)

    # If you build a `Y` that ends at position `j`, then you can register it as
    # looking for a number of items `Z` such that `X -> Y Z`
    #
    # In other words, we can build (0, X/Z, J) and we can initialize it with
    # the rule's weight if we want.
    #wants[j, Z]

    # TODO: is it worth using a sparse agenda on `j`?
    for j in reversed(range(k)):
#    while agenda:
#        j = max(agenda)
#        agenda.remove(j)
#        j = heappop(agenda)
#        j = -j

        for Z in _agenda[j]:

            # TODO: pop in decreasing order of j because that is narrow to wide order
            #(j, Z) = max(agenda)   # TODO: use a priority queue to avoid linear time
            #agenda.remove((j, Z))

            #(j, Z) = agenda.pop()
            z = new[j][Z]

            chart_j = chart[j]
            for i in chart_j:
                chart_ij = chart_j[i]
                if len(chart_ij) == 0: continue

                new_i = new[i]

                for Y, y in chart_ij.items():
                    yz = y * z

                    for r in r_yz_x[Y,Z]:
                        X = r.head
                        x = r.w * yz
                        new_i[X] += x

                        if x != zero:
                            push(i, X)

    return new


def test_cky():
    from arsenal import timers

    _cfg = add_EOS(CFG.from_string("""
    1: S ->  A B
    0.1: A -> A B
    0.4: A ->
    0.5: A -> b
    0.4: B -> a
    0.5: B ->
    0.1: B -> B A
    """, Float))

    cfg = genparse.cfglm.CFGLM(_cfg)

    agenda = AgendaCFGLM(_cfg)

    # brute-force enumerate of the weighted language
    L = _cfg.language(5)

    TIMER = timers()

    all_ok = True
    for x in sorted(L, key=lambda x: (-L[x], x))[:20]:

        with TIMER['agenda']:
            have = agenda(x)

        with TIMER['ordinary']:
            want = _cfg(x)
            #want = L[x]

        err = Float.metric(have, want)
        ok = err <= 1e-4
        all_ok &= ok
        if ok:
            print(colors.mark(ok), repr('⋅'.join(x)), want)
        else:
            print(colors.mark(ok), repr('⋅'.join(x)), colors.red % have, want, 'error', err)
    assert all_ok, [err, have, want]

    TIMER.compare()


def test_new_papa():

    import genparse.examples

    cfg = genparse.examples.papa

    cfglm = genparse.cfglm.CFGLM(cfg)

    agenda = AgendaCFGLM(cfg)

    for prefix in [
            (),
            ('papa',),
            ('papa', 'ate'),
            ('papa', 'ate', 'the'),
            ('papa', 'ate', 'the', 'caviar'),
    ]:
        print()
        print(colors.light.blue % repr(' '.join(prefix)))

        have = agenda.p_next(prefix)
        want = cfglm.p_next(prefix)
        #print(want)
        #print(have)
        print(colors.mark(have.metric(want) <= 1e-8))
        assert have.metric(want) <= 1e-8



def test_generation_speed():

    from genparse import locally_normalize
    from genparse.util import LarkStuff
    from arsenal import timeit, timers
    from arsenal.maths import sample_dict

    lark_stuff = LarkStuff(r"""
    start: WS sum "</s>" | NAME "=" sum "</s>"

    sum: product | sum "+" product | sum MINUS product

    product: atom
        | product "*" atom
        | product "/" atom

    atom: NUMBER
            | MINUS atom
            | NAME
            | "(" sum ")"

    MINUS: /[\-]/
    NUMBER: /[\-+]?\d{1,3}(\.\d{1,3})?/
    WS: /[ \t\f\r\n]/
    NAME: /[a-zA-Z_]{1,5}/
    """)

    with timeit('grammar setup'):
        cfg = locally_normalize(lark_stuff.char_cfg(.999), tol=1e-100).trim()

    #cfg = add_EOS(genparse.examples.papa)

    with timeit('cfglm setup'):
        cfglm = genparse.cfglm.CFGLM(cfg)

    with timeit('agenda setup'):
        agenda = AgendaCFGLM(cfg)


    TIMER = timers()

    for t in range(20):
        prefix = ()
        for _ in range(50):

            print(colors.light.blue % repr(' '.join(prefix)))

            with TIMER['cfglm']:
                want = cfglm.p_next(prefix)

            with TIMER['agenda']:
                have = agenda.p_next(prefix)

            #print(want)
            #print(have)

            y = sample_dict(want)
            prefix = prefix + (y,)

            if y == EOS: break

            #print(colors.mark(have.metric(want) <= 1e-5))
            #print(have.metric(want))

        print(colors.orange % 'final', repr(prefix))
        TIMER.compare()


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
