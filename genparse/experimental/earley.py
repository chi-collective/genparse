from collections import defaultdict
from arsenal.datastructures.pdict import pdict


class Column:
    __slots__ = ('k', 'chart', 'waiting_for', 'Q')

    def __init__(self, k, chart):
        self.k = k
        self.chart = chart
        self.waiting_for = defaultdict(set)
        self.Q = pdict()


class Earley:
    """
    Implements a semiring-weighted version Earley's algorithm that runs in O(N^3|G|) time.
    Warning: Assumes that nullary rules and unary chain cycles have been removed
    """

    __slots__ = ('cfg', 'order', '_chart')

    def __init__(self, cfg):
        assert not cfg.has_nullary() and not cfg.has_unary_cycle()
        self._chart = {}
        self.cfg = cfg
        self.order = cfg._unary_graph_transpose().buckets

    def __call__(self, x):
        N = len(x)

        # return if empty string
        if N == 0:
            return sum(r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ())

        # initialize bookkeeping structures
        self._chart[()] = [Column(0, self.cfg.R.chart())]
        self.PREDICT(self._chart[()][0])

        col = self.chart(x)

        return col[N].chart[0, self.cfg.S]

    def chart(self, x):
        x = tuple(x)
        c = self._chart.get(x)
        if c is None:
            self._chart[x] = c = self._compute_chart(x)
        return c

    def _compute_chart(self, x):
        if len(x) == 0:
            chart = [Column(0, self.cfg.R.chart())]
            self.PREDICT(chart[0])
            return chart
        else:
            chart = self.chart(x[:-1])
            last_chart = self.next_column(chart, x[-1])
            return chart + [last_chart]    # TODO: avoid list addition here as it is not constant time!

    def p_next(self, prefix):
        return self.next_token_weights(self.chart(prefix))

    def next_column(self, prev_cols, token):

        next_col = Column(prev_cols[-1].k + 1, self.cfg.R.chart())

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        for I, X, Ys in prev_cols[-1].waiting_for[token]:
            self._update(next_col, I, X, Ys[1:], prev_cols[-1].chart[I, X, Ys])

        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
        Q = next_col.Q
        while Q:
            (j,Y) = Q.pop()
            col_j = prev_cols[j]
            y = next_col.chart[j,Y]
            for (I, X, Ys) in col_j.waiting_for[Y]:
                self._update(next_col, I, X, Ys[1:], col_j.chart[I,X,Ys] * y)

        self.PREDICT(next_col)

        return next_col

    def PREDICT(self, prev_col):
        # PREDICT: phrase(K, X/Ys, K) += rule(X -> Ys) with lookahead to prune
        k = prev_col.k
        predicted = set()
        for r in self.cfg:
            if r.body == (): continue
            Y = r.body[0]
            item = (k, r.head, r.body)
            was = prev_col.chart[item]
            if was == self.cfg.R.zero:
                prev_col.waiting_for[Y].add(item)
            prev_col.chart[item] = was + r.w

    def _update(self, col, I, X, Ys, value):
        k = col.k
        if Ys == ():
            # Items of the form phrase(I, X/[], K)
            was = col.chart[I,X]
            if was == self.cfg.R.zero:
                col.Q[I,X] = (k if I == k else (k-I-1), self.order[X])
            col.chart[I,X] = was + value

        else:
            # Items of the form phrase(I, X/[Y|Ys], K)
            item = (I, X, Ys)
            was = col.chart[item]
            if was == self.cfg.R.zero:
                col.waiting_for[Ys[0]].add(item)
            col.chart[item] = was + value

    def next_token_weights(self, chart):
        "An O(N²) time algorithm to the total weight of a each next-token extension."

        # set output adjoint to 1; (we drop the empty parens for completed items)
        d_next_col_chart = self.cfg.R.chart()
        d_next_col_chart[0, self.cfg.S] += self.cfg.R.one

        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)

        # loop thru the nodes in reverse order that we popped them
        #next_col = chart[-1]
        #k = next_col.k
        #sort_key = lambda x: (k if x[0] == k else (k-x[0]-1), self.order[x[1]])

        # TODO: we might need case analysis below that distinguishes complete
        # vs. incomplete updates
        #print(self.order)

        # TODO: It should be possible to improve the sparsity in the (j, Y)
        # loops here.  The key is to reverse the order of the forward method.
        for j in range(len(chart)):
            col_j = chart[j]
            for Y in reversed(sorted(self.cfg.N, key=lambda Y: self.order[Y])):
                for (I, X, Ys) in col_j.waiting_for[Y]:

                    # FORWARD PASS:
                    # next_col.chart[I, X, Ys[1:]] += col_j.chart[I,X,Ys] * next_col.chart[j,Y]

                    if len(Ys) != 1: continue
                    d_next_col_chart[j, Y] += col_j.chart[I, X, Ys] * d_next_col_chart[I, X]

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        q = self.cfg.R.chart()
        VV = set(chart[-1].waiting_for)
        VV = VV & self.cfg.V
        for v in VV:
            for I, X, Ys in chart[-1].waiting_for[v]:   # consider all possible tokens here

                # FORWARD PASS:
                # next_col.chart[I, X, Ys[1:]] += prev_cols[-1].chart[I, X, Ys]
                if len(Ys) != 1: continue

                q[v] += chart[-1].chart[I, X, Ys] * d_next_col_chart[I, X]

        return q
