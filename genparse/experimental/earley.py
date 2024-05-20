from collections import defaultdict
from arsenal import colors
from arsenal.datastructures.pdict import pdict
from orderedset import OrderedSet


class Column:
    __slots__ = ('k', 'chart', 'waiting_for', 'Q', 'very_close')

    def __init__(self, k, chart):
        self.k = k
        self.chart = chart
        self.waiting_for = defaultdict(set)
        self.Q = pdict()
        self.very_close = []


class Earley:
    """
    Implements a semiring-weighted version Earley's algorithm that runs in O(N^3|G|) time.
    Warning: Assumes that nullary rules and unary chain cycles have been removed
    """

#    __slots__ = ('cfg', 'order', '_chart')

    def __init__(self, cfg):
        assert not cfg.has_nullary() and not cfg.has_unary_cycle()
        self._chart = {}
        self.cfg = cfg
        self.order = cfg._unary_graph_transpose().buckets


        self.CLOSE = defaultdict(set)


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
        predicted = set()
        for r in self.cfg:
            if r.body == (): continue
            Y = r.body[0]
            item = (k, r.head, r.body)
            was = prev_col.chart[item]
            if was == self.cfg.R.zero:
                prev_col.waiting_for[Y].add(item)

                if len(r.body) == 1:
                    prev_col.very_close.append(item)


                    # very_close[J]((I,X,Ys)) = phrase(I,X/[Y],J)
                    #
                    # (I,X) -> (J,Y)

                    self.CLOSE[k, r.head].add((k, Y))


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
                    col.very_close.append(item)

                    self.CLOSE[I, X].add((col.k, Ys[0]))


            col.chart[item] = was + value

    def next_token_weights(self, chart):
        "An O(N²) time algorithm to the total weight of a each next-token extension."

        # set output adjoint to 1; (we drop the empty parens for completed items)
        d_next_col_chart = self.cfg.R.chart()
        d_next_col_chart[0, self.cfg.S] += self.cfg.R.one

        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)

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

        # TODO: It should be possible to improve the sparsity in the (J, Y)
        # loops here.  The key is to reverse the order of the forward method.

#        for J in range(len(chart)):
##            print(colors.yellow % 'very close:', J, '::', chart[J].very_close)
##
##        #for J in sorted({J for J, Y in chart[-1].C} | {len(chart)-1}):
##        #    Ys = set(chart[J].waiting_for) & self.cfg.N
##            YY = self.cfg.N
##            for Y in reversed(sorted(YY, key=lambda Y: self.order[Y])):
##                for (I, X, Ys) in chart[J].waiting_for[Y]:
#
#            for (I,X,Ys) in sorted(chart[J].very_close, key=lambda item: (-item[0], self.order.get(item[2][0], -1)), reverse=True):
#
#                #assert len(Ys) == 1
#                #assert chart[J].chart[I, X, Ys] != self.cfg.R.zero
#
#                if d_next_col_chart[I, X] == self.cfg.R.zero: continue
#                Y = Ys[0]
#                if self.cfg.is_terminal(Y): continue
#
#                #tmp.append((J,Y,I,X))
##                d_next_col_chart[J, Y] += chart[J].chart[I, X, Ys] * d_next_col_chart[I, X]
#
#                #foo = (I, X, (Y,))
#                #print(colors.mark(foo in chart[J].very_close), foo)
#                #assert foo in chart[J].very_close

        tmp = pdict()
        tmp[0, self.cfg.S] = (0, -self.order[self.cfg.S])
        while tmp:
            (I,X) = tmp.pop()

            for J,Y in self.CLOSE[I,X]:
                if self.cfg.is_terminal(Y): continue

                # TODO: this might be inefficient as it might point to other
                # columns :-/ We might be able to work around this by sorting on J.
                if J >= len(chart): continue

                #tmp.append((J,Y,I,X))
                d_next_col_chart[J, Y] += chart[J].chart[I, X, (Y,)] * d_next_col_chart[I, X]

                tmp[J, Y] = (J, -self.order[Y])


        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        q = self.cfg.R.chart()
        col = chart[-1]
        for I, X, Ys in col.very_close:   # consider all possible tokens here
            if self.cfg.is_nonterminal(Ys[0]): continue

            # FORWARD PASS:
            # next_col.chart[I, X, Ys[1:]] += prev_cols[-1].chart[I, X, Ys]

            q[Ys[0]] += col.chart[I, X, Ys] * d_next_col_chart[I, X]

        return q
