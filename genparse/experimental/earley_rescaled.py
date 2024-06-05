from collections import defaultdict
from arsenal.datastructures.pdict import pdict
from genparse.cfglm import EOS


import numpy as np


from genparse.lm import LM
from genparse import add_EOS, EOS
class EarleyLM(LM):

    def __init__(self, cfg):
        if EOS not in cfg.V: cfg = add_EOS(cfg)
        self.model = Earley(cfg.prefix_grammar)
        super().__init__(V = cfg.V, eos = EOS)

    def p_next(self, context):
        return self.model.p_next(context)

    def __call__(self, context):
        assert context[-1] == EOS
        return self.p_next(context[:-1])[EOS]


class Column:
    __slots__ = ('k', 'chart', 'waiting_for', 'Q', 'very_close_terminal', 'rescale')

    def __init__(self, k, chart):
        self.k = k
        self.chart = chart

        # Within column J, this datastructure maps nonterminals Y to a set of items
        #   Y => {(I, X, Ys) | phrase(I,X/[Y],J) ≠ 0}
        self.waiting_for = defaultdict(set)

        # priority queue used when first filling the column
        self.Q = pdict()

        self.very_close_terminal = []

        self.rescale = None


class Earley:
    """
    Implements a semiring-weighted version Earley's algorithm that runs in O(N^3|G|) time.
    Warning: Assumes that nullary rules and unary chain cycles have been removed
    """

    __slots__ = ('cfg', 'order', '_chart', 'CLOSE', 'V', 'eos', '_initial_column')

    def __init__(self, cfg):

        cfg = cfg.nullaryremove(binarize=True).unarycycleremove().renumber()
        self.cfg = cfg

        # cache of chart columns
        self._chart = {}

        # Topological ordering on the grammar symbols so that we process unary
        # rules in a topological order.
        self.order = cfg._unary_graph_transpose().buckets

        # The `CLOSE` index is used in the `p_next` computation.  It is a data
        # structure implements the following function:
        #
        #   (I,X) => {(J,Y) | phrase(I,X/[Y],J) ≠ 0, Y ∈ cfg.N}
        #
        self.CLOSE = defaultdict(lambda: defaultdict(set))

        col = Column(0, self.cfg.R.chart())
        self.PREDICT(col)
        col.rescale = self.cfg.R.one
        self._initial_column = col

    def __call__(self, x):
        N = len(x)

        # return if empty string
        if N == 0:
            return sum(r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ())

        # initialize bookkeeping structures
        self._chart[()] = [col] = [self._initial_column]

        cols = self.chart(x)

        return cols[N].chart[0, self.cfg.S] / self.rescale(cols, 0, N)

    def rescale(self, cols, I, K):
        "returns the product of the rescaling coefficients for `cols[I:K]`."
        C = self.cfg.R.one
        for c in cols[I:K]:
            C *= c.rescale
        return C

    def log_rescale(self, cols, I, K):
        "returns the product of the rescaling coefficients for `cols[I:K]`."
        return sum(np.log(c.rescale) for c in cols[I:K])

    def chart(self, x):
        x = tuple(x)
        c = self._chart.get(x)
        if c is None:
            self._chart[x] = c = self._compute_chart(x)
        return c

    def _compute_chart(self, x):
        if len(x) == 0:
            return [self._initial_column]
        else:
            chart = self.chart(x[:-1])
            last_chart = self.next_column(chart, x[-1])
            return chart + [last_chart]    # TODO: avoid list addition here as it is not constant time!

    def p_next(self, prefix):
        return self.next_token_weights(self.chart(prefix))

    def logp(self, x):
        cols = self.chart(x)
        N = len(x)
        return np.log(cols[N].chart[0, self.cfg.S]) - self.log_rescale(cols, 0, N)

    def next_column(self, prev_cols, token):

        next_col = Column(prev_cols[-1].k + 1, self.cfg.R.chart())

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        prev_col = prev_cols[-1]
        for I, X, Ys in prev_col.waiting_for[token]:
            self._update(next_col, I, X, Ys[1:], prev_col.chart[I, X, Ys] * prev_col.rescale)

        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
        Q = next_col.Q
        while Q:
            (J,Y) = Q.pop()
            col_J = prev_cols[J]
            y = next_col.chart[J,Y]
            for (I, X, Ys) in col_J.waiting_for[Y]:
                self._update(next_col, I, X, Ys[1:], col_J.chart[I,X,Ys] * y)

        self.PREDICT(next_col)

        k = next_col.k

        num = prev_col.chart[0, self.cfg.S]
        den = next_col.chart[0, self.cfg.S]

        if den == 0 or num == 0:
            next_col.rescale = 1
        else:
            next_col.rescale = num / den * prev_col.rescale

        return next_col

    def PREDICT(self, prev_col):
        # PREDICT: phrase(K, X/Ys, K) += rule(X -> Ys) with lookahead to prune
        k = prev_col.k
        zero = self.cfg.R.zero
        for r in self.cfg:
            if r.body == (): continue
            Y = r.body[0]
            item = (k, r.head, r.body)
            was = prev_col.chart[item]
            if was == zero:
                prev_col.waiting_for[Y].add(item)

                if len(r.body) == 1:
                    if self.cfg.is_terminal(Y):
                        prev_col.very_close_terminal.append(item)
                    else:
                        # very_close[J]((I,X,Ys)) = phrase(I,X/[Y],J)
                        #
                        # (I,X) -> (J,Y)
                        self.CLOSE[k, r.head][k].add(Y)

            prev_col.chart[item] = was + r.w

    def _update(self, col, I, X, Ys, value):
        K = col.k
        if Ys == ():
            # Items of the form phrase(I, X/[], K)
            was = col.chart[I,X]
            if was == self.cfg.R.zero:
                col.Q[I,X] = (K if I == K else (K-I-1), self.order[X])
            col.chart[I,X] = was + value

        else:
            # Items of the form phrase(I, X/[Y|Ys], K)
            item = (I, X, Ys)
            was = col.chart[item]
            if was == self.cfg.R.zero:
                col.waiting_for[Ys[0]].add(item)

                if len(Ys) == 1:
                    Y = Ys[0]
                    if self.cfg.is_terminal(Y):
                        col.very_close_terminal.append(item)
                    else:
                        self.CLOSE[I, X][col.k].add(Y)

            col.chart[item] = was + value

    # We have derived the `next_token_weights` algorithm by backpropagation on
    # the program with respect to the item `phrase(0, s, K)`.
    #
    # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
    #
    # Directly applying the gradient transformation, we get
    #
    # ∇phrase(0, s/[], K) += 1
    # ∇phrase(J, Y/[], K) += phrase(I, X/[Y|Ys], J) * ∇phrase(I, X/Ys, K)
    #
    # Some quick analysis reveals that the `Ys` list must always be [], and
    # that K is always equal to the final column.  We specialize the program
    # below:
    #
    # ∇phrase(0, s/[], K) += 1
    # ∇phrase(J, Y/[], K) += phrase(I, X/[Y], J) * ∇phrase(I, X/[], K)
    #
    # We can abbreviate the names:
    #
    # q(0, s) += 1
    # q(J, Y) += phrase(I, X/[Y], J) * q(I, X)
    #
    # These items satisfy (I > J) and (X > Y) where the latter is the
    # nonterminal ordering.

    def next_token_weights(self, chart):
        "An O(N²) time algorithm to the total weight of a each next-token extension."

        N = len(chart)
        order = self.order
        CLOSE = self.CLOSE

        # XXX: the rescaling coefficient will cancel out when we normalized the next-token weights
        #C = self.rescale(chart, 0, N-1)

        # set output adjoint to 1; (we drop the empty parens for completed items)
        d_next_col_chart = self.cfg.R.chart()
        d_next_col_chart[0, self.cfg.S] += self.cfg.R.one #/ C

        tmp_J = pdict()
        tmp_J_Y = [pdict() for _ in range(N)]

        tmp_J[0] = 0

        tmp_J_Y[0][self.cfg.S] = -order[self.cfg.S]

        zero = self.cfg.R.zero

        while tmp_J:
            I = tmp_J.pop()

            xxx = tmp_J_Y[I]

            #already_popped = set()
            while xxx:

                X = xxx.pop()

                #assert X not in already_popped
                #already_popped.add(X)

                value = d_next_col_chart[I, X]

                #assert value != zero

                close_IX = CLOSE[I,X]

                for J in sorted(close_IX):   # TODO: more efficient to maintain sorted?

                    if J >= N:
                        break

                    chart_J_chart = chart[J].chart

                    yyy = tmp_J_Y[J]
                    pushed = False
                    for Y in close_IX[J]:

                        #if self.cfg.is_terminal(Y): continue
                        #assert self.cfg.is_nonterminal(Y)

                        #tmp.append((J,Y,I,X))
                        new_value = chart_J_chart[I, X, (Y,)] * value
                        if new_value != zero:
                            d_next_col_chart[J, Y] += new_value

                            yyy[Y] = -order[Y]
                            pushed = True

                    if pushed:
                        tmp_J[J] = J

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        q = self.cfg.R.chart()
        col = chart[-1]
        for I, X, Ys in col.very_close_terminal:   # consider all possible tokens here
            #assert self.cfg.is_nonterminal(Ys[0])

            # FORWARD PASS:
            # next_col.chart[I, X, Ys[1:]] += prev_cols[-1].chart[I, X, Ys]

            q[Ys[0]] += col.chart[I, X, Ys] * d_next_col_chart[I, X]

        return q.normalize()
