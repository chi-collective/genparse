from collections import defaultdict
from arsenal.datastructures.pdict import pdict
from genparse.cfglm import EOS
from functools import lru_cache

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
    __slots__ = ('k', 'chart', 'waiting_for', 'Q')

    def __init__(self, k, chart):
        self.k = k
        self.chart = chart

        # Within column J, this datastructure maps nonterminals Y to a set of items
        #   Y => {(I, X, Ys) | phrase(I,X/[Y],J) ≠ 0}
        self.waiting_for = defaultdict(set)

        # priority queue used when first filling the column
        self.Q = pdict()


class Earley:
    """
    Implements a semiring-weighted version Earley's algorithm that runs in O(N^3|G|) time.
    Warning: Assumes that nullary rules and unary chain cycles have been removed
    """

    __slots__ = ('cfg', 'order', '_chart', 'V', 'eos', '_initial_column')

    def __init__(self, cfg):

        cfg = cfg.nullaryremove(binarize=True).unarycycleremove().renumber()
        self.cfg = cfg

        # cache of chart columns
        self._chart = {}

        # Topological ordering on the grammar symbols so that we process unary
        # rules in a topological order.
        self.order = cfg._unary_graph_transpose().buckets

        col = Column(0, self.cfg.R.chart())
        self.PREDICT(col)
        self._initial_column = col

    def __call__(self, x):
        N = len(x)

        # return if empty string
        if N == 0:
            return sum(r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ())

        # initialize bookkeeping structures
        self._chart[()] = [col] = [self._initial_column]

        cols = self.chart(x)

        return cols[N].chart[0, self.cfg.S]

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

    def next_column(self, prev_cols, token):

        next_col = Column(prev_cols[-1].k + 1, self.cfg.R.chart())

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        prev_col = prev_cols[-1]
        for I, X, Ys in prev_col.waiting_for[token]:
            self._update(next_col, I, X, Ys[1:], prev_col.chart[I, X, Ys])

        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
        Q = next_col.Q
        while Q:
            (J,Y) = Q.pop()
            col_J = prev_cols[J]
            y = next_col.chart[J,Y]
            for (I, X, Ys) in col_J.waiting_for[Y]:
                self._update(next_col, I, X, Ys[1:], col_J.chart[I,X,Ys] * y)

        self.PREDICT(next_col)

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
    #
    # Failed Idea: Let's transpose the program by LCT
    #
    # (d(W) / q(I,X)) += phrase(I,X / [Y],J) * d(W) / q(J,Y).
    # (d(W) / q(I,X)) += phrase(I,X / [W],J) * len(J) * terminal(W).
    # d(W) += d(W) / q(0, s).
    #
    # unforuntately, this leads to a slower program as it is based on forward-mode differentiation.
    #
    # d(W) += g(W,0,s).
    # g(W,I,X) += phrase(I,X / [Y],J) * g(W,J,Y).
    # g(W,I,X) += phrase(I,X / [W],J) * len(J) * terminal(W).
    #
#    def next_token_weights(self, chart):
#        "An O(N²) time algorithm to the total weight of a each next-token extension."
#
#        N = len(chart)
#        order = self.order
#        #CLOSE = self.CLOSE
#
#
#
#        # The `CLOSE` index is used in the `p_next` computation.  It is a data
#        # structure implementing the following function:
#        #
#        #   (I,X) => {(J,Y) | phrase(I,X/[Y],J) ≠ 0, Y ∈ cfg.N}
#        #
#        CLOSE = defaultdict(lambda: defaultdict(set))
#        very_close_terminal = []
#
#        for col in chart:
#            for item in col.chart:
#                if len(item) != 3: continue
#                (I, X, Ys) = item
#                if len(Ys) == 1:
#                    Y = Ys[0]
#                    if self.cfg.is_terminal(Y):
#                        if col.k == N-1:
#                            very_close_terminal.append(item)
#                    else:
#                        CLOSE[I, X][col.k].add(Y)
#
#        # set output adjoint to 1; (we drop the empty parens for completed items)
#        d_next_col_chart = self.cfg.R.chart()
#        d_next_col_chart[0, self.cfg.S] += self.cfg.R.one
#
#        tmp_J = pdict()
#        tmp_J_Y = [pdict() for _ in range(N)]
#
#        tmp_J[0] = 0
#
#        tmp_J_Y[0][self.cfg.S] = -order[self.cfg.S]
#
#        zero = self.cfg.R.zero
#
#        while tmp_J:
#            I = tmp_J.pop()
#
#            xxx = tmp_J_Y[I]
#
#            #already_popped = set()
#            while xxx:
#
#                X = xxx.pop()
#
#                #assert X not in already_popped
#                #already_popped.add(X)
#
#                value = d_next_col_chart[I, X]
#
#                #assert value != zero
#
#                close_IX = CLOSE[I, X]
#
#                for J in sorted(close_IX):   # TODO: more efficient to maintain sorted?
#
#                    if J >= N:
#                        break
#
#                    chart_J_chart = chart[J].chart
#
#                    yyy = tmp_J_Y[J]
#                    pushed = False
#                    for Y in close_IX[J]:
#
#                        #if self.cfg.is_terminal(Y): continue
#                        #assert self.cfg.is_nonterminal(Y)
#
#                        #tmp.append((J,Y,I,X))
#                        new_value = chart_J_chart[I, X, (Y,)] * value
#                        if new_value != zero:
#                            d_next_col_chart[J, Y] += new_value
#
#                            yyy[Y] = -order[Y]
#                            pushed = True
#
#                    if pushed:
#                        tmp_J[J] = J
#
#        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
#        q = self.cfg.R.chart()
#        col = chart[-1]
#        for I, X, Ys in very_close_terminal:   # consider all possible tokens here
#            #assert self.cfg.is_nonterminal(Ys[0])
#
#            # FORWARD PASS:
#            # next_col.chart[I, X, Ys[1:]] += prev_cols[-1].chart[I, X, Ys]
#
#            q[Ys[0]] += col.chart[I, X, Ys] * d_next_col_chart[I, X]
#
#        return q

#    def next_token_weights(self, chart):
#        "An O(N²) time algorithm to the total weight of a each next-token extension."
#
#        # set output adjoint to 1; (we drop the empty parens for completed items)
#        d_next_col_chart = self.cfg.R.chart()
#        d_next_col_chart[0, self.cfg.S] += self.cfg.R.one
#
#        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
#
#        # TODO: It should be possible to improve the sparsity in the (j, Y)
#        # loops here.  The key is to reverse the order of the forward method.
#        for J in range(len(chart)):
#            col_J = chart[J]
#            for Y in reversed(sorted(self.cfg.N, key=lambda Y: self.order[Y])):
#                for (I, X, Ys) in col_J.waiting_for[Y]:
#                    if len(Ys) != 1: continue
#                    if col_J.chart[I, X, Ys] == self.cfg.R.zero: continue
#
#                    # FORWARD PASS:
#                    # next_col[I, X, Ys[1:]] += col_j.chart[I,X,Ys] * next_col.chart[J,Y]
#
#                    d_next_col_chart[J, Y] += col_J.chart[I, X, Ys] * d_next_col_chart[I, X]
#
#        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
#        q = self.cfg.R.chart()
#        prev_col = chart[-1]
#        VV = set(prev_col.waiting_for)
#        VV = VV & self.cfg.V
#        for v in VV:
#            for I, X, Ys in prev_col.waiting_for[v]:   # consider all possible tokens here
#                if len(Ys) != 1: continue
#
#                # FORWARD PASS:
#                # next_col.chart[I, X, Ys[1:]] += prev_cols[-1].chart[I, X, Ys]
#
#                q[v] += prev_col.chart[I, X, Ys] * d_next_col_chart[I, X]
#
#        return q

    def next_token_weights(self, chart):

        # Let's try a backward chaining algorithm
        #
        # q(0, s) += 1
        # q(J, Y) += phrase(I, X/[Y], J) * q(I, X)
        #
        # p(W) += q(I, X) * phrase(I, X/[W], J)  where len(J) * terminal(W).
        #

        @lru_cache
        def q(J, Y):
            if J == 0 and Y == self.cfg.S:
                return self.cfg.R.one
            else:
                result = self.cfg.R.zero
                for (I, X, Ys) in chart[J].waiting_for[Y]:
                    if len(Ys) == 1:
                        result += chart[J].chart[I, X, Ys] * q(I, X)
                return result

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        p = self.cfg.R.chart()
        col = chart[-1]
        for item in col.chart:
            if len(item) == 2: continue

            (I, X, Ys) = item

            if len(Ys) == 1 and self.cfg.is_terminal(Ys[0]):

                p[Ys[0]] += col.chart[I, X, Ys] * q(I, X)

        return p
