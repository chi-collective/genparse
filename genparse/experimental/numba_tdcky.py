"""
Experimental tensor decomposition method for implementing CKY.

"""
import numpy as np
from genparse import CFG, Float, add_EOS, EOS
import genparse.examples
from arsenal import colors
from collections import defaultdict
from arsenal import Integerizer

# https://github.com/google-deepmind/synjax/blob/master/synjax/_src/constituency_tensor_decomposition_pcfg.py

# https://aclanthology.org/2022.naacl-main.353.pdf
# https://github.com/sustcsonglin/TN-PCFG/blob/main/parser/pcfgs/tdpcfg.py

from numba import jit


class ExactTensorDecomp:

    def __init__(self, cfg):

        cfg = cfg.cnf.renumber()
        (nullary, terminal, binary) = cfg._cnf

        # f is the rank space encoding
        f = Integerizer()
        for r in binary:
            f.add(r.body)

        #print('|x,y|', len({(r.head, r.body[0]) for r in binary}))
        #print('|x,z|', len({(r.head, r.body[1]) for r in binary}))
        #print('|y,z|', len({(r.body[0], r.body[1]) for r in binary}))
        #print('|x,y,z|', len({(r.head, r.body[0], r.body[1]) for r in binary}))

        Rank = len(f)
        NT = len(cfg.N)
        x = np.zeros((Rank, NT))
        y = np.zeros((Rank, NT))
        z = np.zeros((Rank, NT))

        for R, r in enumerate(binary):   # TODO: there are better factorizations!
            X,[Y,Z] = r.head, r.body
            R = f((Y,Z))
            y[R,Y] = 1
            z[R,Z] = 1
            x[R,X] += r.w

        self.x = x
        self.y = y
        self.z = z
        self.S = cfg.S
        self.nullary = nullary
        self.terminal = terminal
        self.f = f
        self.cfg = cfg
        self.NT = NT
        self.Rank = Rank

        self._chart = {}

    def __call__(self, xs):
        (b, _, _) = self.chart(xs)
        return b[len(xs)][0][self.S]

    def p_next(self, prefix):
        (b, by, bz) = self.chart(prefix)
        return self.next_token_weights(b, by, bz, prefix)

    def chart(self, prefix):
        c = self._chart.get(prefix)
        if c is None:
            c = self._compute_chart(prefix)
            self._chart[prefix] = c
        return c

    def _compute_chart(self, prefix):
        if len(prefix) == 0:
            (b_0, by_0, bz_0) = self.extend_chart(prefix, None, None, None)
            return (
                [b_0],
                [by_0],
                [bz_0],
            )
        else:
            (b, by, bz) = self.chart(prefix[:-1])
            (b_K, by_K, bz_K) = self.extend_chart(prefix, b, by, bz)
            return (
                b + [b_K],
                by + [by_K],
                bz + [bz_K],
            )

    def extend_chart(self, xs, b, by, bz):
        K = len(xs)
        x = self.x
        y = self.y
        z = self.z
        Rank = self.Rank
        NT = self.NT

        if K == 0:
            # nullary rule

            b_K = np.zeros((K+1, NT))
            by_K = np.zeros((K+1, Rank))
            bz_K = np.zeros((K+1, Rank))

            b_K[K, self.S] += self.nullary
            return (b_K, by_K, bz_K)

        terminal_xs_K = np.zeros(NT)
        if K > 0:
            for r in self.terminal[xs[K-1]]:
                terminal_xs_K[r.head] += r.w

        return _extend_chart(x=x,
                             y=y,
                             z=z,
                             S=self.S,
                             K=K,
                             Rank=Rank,
                             NT=NT,
                             nullary=self.nullary,
                             terminal_xs_K=terminal_xs_K,
                             b=b,
                             by=by,
                             bz=bz)

    def next_token_weights(self, b, by, bz, prefix):

        K = len(prefix) + 1

        D_b_K = _next_token_weights(
            x = self.x,
            y = self.y,
            z = self.z,
            b = b,
            by = by,
            bz = bz,
            S = self.S,
            K = K,
        )

        q = Float.chart()
        for w in self.cfg.V:

            #for X in range(NT):
            #    for R in range(Rank):
            #        by_K[I][R] += y[R,X] * b_K[I][X]
            #        bz_K[I][R] += z[R,X] * b_K[I][X]

            for r in self.terminal[w]:
                q[w] += r.w * D_b_K[K-1][r.head]

        return q


def _extend_chart(
        x,
        y,
        z,

        S: int,
        K: int,
        Rank: int,
        NT: int,

        nullary,
        terminal_xs_K,
        b: list,
        by: list,
        bz: list,
):

    b_K = np.zeros((K+1, NT))
    by_K = np.zeros((K+1, Rank))
    bz_K = np.zeros((K+1, Rank))

    if K == 0:
        # nullary rule
#        b_K[K] = np.zeros(NT)
        b_K[K,S] += nullary

    if K > 0:
        # preterminal rules
        I = K-1
        b_K[I] = np.zeros(NT)
        b_K[I] += terminal_xs_K
        by_K[I] = y @ b_K[I]
        bz_K[I] = z @ b_K[I]

    tmp = np.zeros(Rank, dtype=float)

    for span in range(2, K+1):
        I = K - span

        # this is batched matrix multiplication
        tmp[:] = 0
        for J in range(I + 1, K):
            foo = by[J]
            for R in range(Rank):
                tmp[R] += foo[I,R] * bz_K[J,R]

        b_K[I] = tmp @ x
        by_K[I] = y @ b_K[I]
        bz_K[I] = z @ b_K[I]

    return (b_K, by_K, bz_K)


@jit
def _next_token_weights(x, y, z, b, by, bz, S, K):

    Rank, NT = x.shape

    D_b_K = np.zeros((K, NT))
    D_by_K = np.zeros((K, Rank))
    D_bz_K = np.zeros((K, Rank))

    D_b_K[0][S] += 1

    b_K = b[K-1]
    by_K = by[K-1]
    bz_K = bz[K-1]

    for span in range(K, 1, -1):
        I = K - span

        #for X in range(NT):
        #    for R in range(Rank)
        #        by_K[I][R] = y[R,X] * b_K[I][X]
        #        bz_K[I][R] = z[R,X] * b_K[I][X]

        for R in range(Rank):
            for X in range(NT):
                D_b_K[I][X] += z[R,X] * D_bz_K[I][R]
                D_b_K[I][X] += y[R,X] * D_by_K[I][R]

        #for R in range(Rank):
        #    for X in range(NT):
        #        b_K[I][X] = tmp[R] * x[R,X]

        D_tmp = np.zeros(Rank)
        for R in range(Rank):
            for X in range(NT):
                D_tmp[R] += D_b_K[I][X] * x[R,X]

        #tmp = np.zeros(Rank)
        #for J in range(I + 1, K):
        #    for R in range(Rank):
        #        tmp[R] += by[J][I][R] * bz_K[J][R]

        for J in range(I + 1, K):
            for R in range(Rank):
                D_bz_K[J][R] += by[J][I][R] * D_tmp[R]

    # Preterminal
    I = K-1

    for X in range(NT):
        for R in range(Rank):
            D_b_K[I][X] += y[R,X] * D_by_K[I][R]
            D_b_K[I][X] += z[R,X] * D_bz_K[I][R]

    return D_b_K


def test_cky():

    cfg = CFG.from_string("""
    1: S ->  A B
    0.1: A -> A B
    0.4: A ->
    0.5: A -> b
    0.4: B -> a
    0.5: B ->
    0.1: B -> B A
    """, Float)

    cfg = cfg.cnf.renumber()

    # JIT warm up......
    decomp = ExactTensorDecomp(cfg)
    L = cfg.language(4)
    for x in sorted(L, key=lambda x: (-L[x], x))[:20]:
        have = decomp(x)

#    import genparse.cfglm
#    prefix = 'bab'
#    b, by, bz = decomp.chart(prefix)
#    q = decomp.next_token_weights(b, by, bz, prefix)
#    print('have=', q)

#    want = genparse.cfglm.CFGLM(cfg).p_next(prefix)
#    print('want=', want)


    decomp = ExactTensorDecomp(cfg)

    # brute-force enumerate of the weighted language
    L = cfg.language(4)

    from arsenal import timers
    TIMER = timers()

    all_ok = True
    for x in sorted(L, key=lambda x: (-L[x], x))[:20]:

        with TIMER['td']:
            have = decomp(x)

        with TIMER['ordinary']:
            want = cfg(x)

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

    from genparse import CFGLM

    cfg = add_EOS(genparse.examples.papa)

    cfglm = CFGLM(cfg)

    decomp = ExactTensorDecomp(cfg.prefix_grammar)

    for prefix in [
            (),
            ('papa',),
            ('papa', 'ate'),
            ('papa', 'ate', 'the'),
            ('papa', 'ate', 'the', 'caviar'),
    ]:
        print()
        print(colors.light.blue % repr(' '.join(prefix)))

        have = decomp.p_next(prefix)
        want = cfglm.p_next(prefix)
        #print(want)
        #print(have)
        print(colors.mark(have.metric(want) <= 1e-5))
        assert have.metric(want) <= 1e-5



def test_generation_speed():

    from genparse import CFGLM, locally_normalize
    from genparse.util import LarkStuff
    from arsenal import timeit, timers
    from arsenal.maths import sample_dict

    #from genparse.tdcky import ExactTensorDecomp

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
        # the base character-level CFG language model
        cfglm = CFGLM(cfg)

    with timeit('decomp setup'):
        decomp = ExactTensorDecomp(cfglm.pfg)

    TIMER = timers()

    for t in range(20):
        prefix = ()
        for _ in range(20):
            #print(colors.light.blue % repr(' '.join(prefix)))

            with TIMER['cfglm']:
                want = cfglm.p_next(prefix)

            if t == 0:
                have = decomp.p_next(prefix)
            else:
                with TIMER['decomp']:
                    have = decomp.p_next(prefix)

            y = sample_dict(want)
            prefix = prefix + (y,)

            if y == EOS: break

            #print(want)
            #print(have)
            #print(colors.mark(have.metric(want) <= 1e-5))
            #print(have.metric(want))

        print(colors.orange % 'final', repr(prefix))
        TIMER.compare()


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
