"""
Experimental tensor decomposition method for implementing CKY.

"""

import numpy as np
from arsenal import Integerizer, colors

import genparse.examples
from genparse.cfg import CFG
from genparse.cfglm import add_EOS
from genparse.semiring import Float

# https://github.com/google-deepmind/synjax/blob/master/synjax/_src/constituency_tensor_decomposition_pcfg.py

# https://aclanthology.org/2022.naacl-main.353.pdf
# https://github.com/sustcsonglin/TN-PCFG/blob/main/parser/pcfgs/tdpcfg.py


class ExactTensorDecomp:

    def __init__(self, cfg):

        cfg = cfg.cnf.renumber()
        (nullary, terminal, binary) = cfg._cnf

        # f is the rank space encoding
        f = Integerizer()
        for r in binary:
            f.add(r.body)

        # print('|x,y|', len({(r.head, r.body[0]) for r in binary}))
        # print('|x,z|', len({(r.head, r.body[1]) for r in binary}))
        # print('|y,z|', len({(r.body[0], r.body[1]) for r in binary}))
        # print('|x,y,z|', len({(r.head, r.body[0], r.body[1]) for r in binary}))

        Rank = len(f)
        NT = len(cfg.N)
        x = np.zeros((Rank, NT))
        y = np.zeros((Rank, NT))
        z = np.zeros((Rank, NT))

        for R, r in enumerate(binary):  # TODO: there are better factorizations!
            X, [Y, Z] = r.head, r.body
            R = f((Y, Z))
            y[R, Y] = 1
            z[R, Z] = 1
            x[R, X] += r.w

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
        (b, _) = self.chart(xs)
        return b[len(xs)][0][self.S]

    def p_next(self, prefix):
        (_, by) = self.chart(prefix)
        return self.next_token_weights(by, prefix)

    def chart(self, prefix):
        c = self._chart.get(prefix)
        if c is None:
            c = self._compute_chart(prefix)
            self._chart[prefix] = c
        return c

    def _compute_chart(self, prefix):
        if len(prefix) == 0:
            (b_0, by_0) = self.extend_chart(prefix, None)
            return (
                [b_0],
                [by_0],
            )
        else:
            (b, by) = self.chart(prefix[:-1])
            (b_K, by_K) = self.extend_chart(prefix, by)
            return (
                b + [b_K],
                by + [by_K],
            )

    def extend_chart(self, xs, by):

        K = len(xs)
        x = self.x
        y = self.y
        z = self.z
        Rank = self.Rank
        NT = self.NT
        nullary = self.nullary
        terminal = self.terminal

        b_K = np.zeros((K + 1, NT))
        by_K = np.zeros((K + 1, Rank))
        bz_K = np.zeros((K + 1, Rank))

        if K == 0:
            # nullary rule
            b_K[K] = np.zeros(NT)
            b_K[K][self.S] += nullary

        if K > 0:
            # preterminal rules
            I = K - 1
            b_K[I] = np.zeros(NT)
            for r in terminal[xs[K - 1]]:
                b_K[I][r.head] += r.w
            by_K[I] = y @ b_K[I]
            bz_K[I] = z @ b_K[I]

        for span in range(2, K + 1):
            I = K - span

            # this is batched matrix multiplication
            tmp = np.zeros(Rank)
            for J in range(I + 1, K):
                tmp += by[J][I] * bz_K[J]

            b_K[I] = tmp @ x
            by_K[I] = y @ b_K[I]
            bz_K[I] = z @ b_K[I]

        return (b_K, by_K)

    # TODO: vectorize
    def next_token_weights(self, by, prefix):  # XXX: why is _b unused?
        K = len(prefix) + 1  # change to upper case to match the forward pass

        x = self.x
        y = self.y
        z = self.z
        Rank, NT = x.shape
        terminal = self.terminal

        D_b_K = np.zeros((K, NT))
        D_by_K = np.zeros((K, Rank))
        D_bz_K = np.zeros((K, Rank))

        D_b_K[0][self.S] += 1

        for span in reversed(range(2, K + 1)):
            I = K - span

            # for X in range(NT):
            #    for R in range(Rank)
            #        by_K[I][R] = y[R,X] * b_K[I][X]
            #        bz_K[I][R] = z[R,X] * b_K[I][X]

            for R in range(Rank):
                for X in range(NT):
                    D_b_K[I][X] += z[R, X] * D_bz_K[I][R]
                    D_b_K[I][X] += y[R, X] * D_by_K[I][R]

            # for R in range(Rank):
            #    for X in range(NT):
            #        b_K[I][X] = tmp[R] * x[R,X]

            D_tmp = np.zeros(Rank)
            for R in range(Rank):
                for X in range(NT):
                    D_tmp[R] += D_b_K[I][X] * x[R, X]

            # tmp = np.zeros(Rank)
            # for J in range(I + 1, K):
            #    for R in range(Rank):
            #        tmp[R] += by[J][I][R] * bz_K[J][R]

            for J in range(I + 1, K):
                for R in range(Rank):
                    D_bz_K[J][R] += by[J][I][R] * D_tmp[R]

        # Preterminal
        q = self.cfg.R.chart()

        I = K - 1

        for X in range(NT):
            for R in range(Rank):
                D_b_K[I][X] += y[R, X] * D_by_K[I][R]
                D_b_K[I][X] += z[R, X] * D_bz_K[I][R]

        for w in self.cfg.V:

            # for X in range(NT):
            #    for R in range(Rank):
            #        by_K[I][R] += y[R,X] * b_K[I][X]
            #        bz_K[I][R] += z[R,X] * b_K[I][X]

            for r in terminal[w]:
                q[w] += r.w * D_b_K[K - 1][r.head]

        return q


def test_cky():

    cfg = CFG.from_string(
        """
    1: S ->  A B
    0.1: A -> A B
    0.4: A ->
    0.5: A -> b
    0.4: B -> a
    0.5: B ->
    0.1: B -> B A
    """,
        Float,
    )

    cfg = cfg.cnf.renumber()

    decomp = ExactTensorDecomp(cfg)

    #    import genparse.cfglm
    #    prefix = 'bab'
    #    b, by, bz = decomp.chart(prefix)
    #    q = decomp.next_token_weights(by, prefix)
    #    print('have=', q)

    #    want = genparse.cfglm.CFGLM(cfg).p_next(prefix)
    #    print('want=', want)

    # brute-force enumerate of the weighted language
    L = cfg.language(4)

    from arsenal import timers

    TIMER = timers()

    all_ok = True
    for x in sorted(L, key=lambda x: (-L[x], x))[:20]:

        with TIMER["td"]:
            have = decomp(x)

        with TIMER["ordinary"]:
            want = cfg(x)

        err = Float.metric(have, want)
        ok = err <= 1e-4
        all_ok &= ok
        if ok:
            print(colors.mark(ok), repr("⋅".join(x)), want)
        else:
            print(
                colors.mark(ok),
                repr("⋅".join(x)),
                colors.red % have,
                want,
                "error",
                err,
            )
    assert all_ok, [err, have, want]

    TIMER.compare()


def test_new_papa():

    from genparse import CFGLM

    cfg = add_EOS(genparse.examples.papa)

    cfglm = CFGLM(cfg)

    decomp = ExactTensorDecomp(cfg.prefix_grammar)

    for prefix in [
        (),
        ("papa",),
        ("papa", "ate"),
        ("papa", "ate", "the"),
        ("papa", "ate", "the", "caviar"),
    ]:
        print()
        print(colors.light.blue % repr(" ".join(prefix)))

        (_, by) = decomp.chart(prefix)
        have = decomp.next_token_weights(by, prefix)

        want = cfglm.p_next(prefix)
        print(want)
        print(have)
        print(colors.mark(have.metric(want) <= 1e-5))
        assert have.metric(want) <= 1e-5


if __name__ == "__main__":
    from arsenal import testing_framework

    testing_framework(globals())
