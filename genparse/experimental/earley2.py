from collections import defaultdict
from functools import lru_cache

from arsenal.datastructures.pdict import pdict

from genparse.cfglm import EOS, add_EOS
from genparse.linear import WeightedGraph
from genparse.lm import LM
from genparse.semiring import Boolean


class EarleyLM(LM):

    def __init__(self, cfg):
        if EOS not in cfg.V:
            cfg = add_EOS(cfg)
        self.model = Earley(cfg.prefix_grammar)
        super().__init__(V=cfg.V, eos=EOS)

    def p_next(self, context):
        return self.model.p_next(context)

    def __call__(self, context):
        assert context[-1] == EOS
        return self.p_next(context[:-1])[EOS]


class Column:
    __slots__ = ("k", "i_chart", "c_chart", "waiting_for", "Q")

    def __init__(self, k):
        self.k = k
        self.i_chart = {}
        self.c_chart = {}

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

    __slots__ = ("cfg", "order", "_chart", "V", "eos", "_initial_column", "R")

    def __init__(self, cfg):

        cfg = cfg.nullaryremove(binarize=True).unarycycleremove().renumber()
        self.cfg = cfg

        # cache of chart columns
        self._chart = {}

        # Topological ordering on the grammar symbols so that we process unary
        # rules in a topological order.
        self.order = cfg._unary_graph_transpose().buckets

        # left-corner graph
        R = WeightedGraph(Boolean)
        for r in cfg:
            if len(r.body) == 0:
                continue
            A = r.head
            B = r.body[0]
            R[A, B] += Boolean.one
        self.R = R

        col = Column(0)
        self.PREDICT(col)
        self._initial_column = col

    def clear_cache(self):
        self._chart.clear()

    def __call__(self, x):
        N = len(x)

        # return if empty string
        if N == 0:
            return sum(r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ())

        # initialize bookkeeping structures
        self._chart[()] = [col] = [self._initial_column]

        cols = self.chart(x)

        value = cols[N].c_chart.get((0, self.cfg.S))
        return value if value is not None else self.cfg.R.zero

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
            return chart + [
                last_chart
            ]  # TODO: avoid list addition here as it is not constant time!

    def p_next(self, prefix):
        return self.next_token_weights(self.chart(prefix))

    def next_column(self, prev_cols, token):

        next_col = Column(prev_cols[-1].k + 1)

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        prev_col = prev_cols[-1]
        for item in prev_col.waiting_for[token]:
            (I, X, Ys) = item
            self._update(next_col, I, X, Ys[1:], prev_col.i_chart[item])

        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
        Q = next_col.Q
        while Q:
            (J, Y) = item = Q.pop()
            col_J = prev_cols[J]
            y = next_col.c_chart[item]
            for item in col_J.waiting_for[Y]:
                (I, X, Ys) = item
                self._update(next_col, I, X, Ys[1:], col_J.i_chart[item] * y)

        self.PREDICT(next_col)

        return next_col

    def PREDICT(self, prev_col):
        # PREDICT: phrase(K, X/Ys, K) += rule(X -> Ys) with some filtering heuristics
        k = prev_col.k
        prev_col_chart = prev_col.i_chart
        prev_col_waiting_for = prev_col.waiting_for

        # Filtering heuristic: Don't create the predicted item (K, X, [...], K)
        # unless there exists an item that wants the X item that it may
        # eventually provide.  In other words, for predicting this item to be
        # useful there must be an item of the form (I, X', [X, ...], K) in this
        # column for which lc(X', X) is true.
        if prev_col.k == 0:
            targets = {self.cfg.S}
        else:
            targets = set(prev_col.waiting_for)

        reachable = set(targets)
        agenda = list(targets)
        while agenda:
            X = agenda.pop()
            for Y in self.R.outgoing[X]:
                if Y not in reachable:
                    reachable.add(Y)
                    agenda.append(Y)

        for X in reachable:
            for r in self.cfg.rhs[X]:
                Ys = r.body
                if Ys == ():
                    continue
                item = (k, X, Ys)
                was = prev_col_chart.get(item)
                if was is None:
                    Y = Ys[0]
                    prev_col_waiting_for[Y].add(item)
                    prev_col_chart[item] = r.w
                else:
                    prev_col_chart[item] = was + r.w

    def _update(self, col, I, X, Ys, value):
        K = col.k
        if Ys == ():
            # Items of the form phrase(I, X/[], K)
            item = (I, X)
            was = col.c_chart.get(item)
            if was is None:
                col.Q[item] = (K if I == K else (K - I - 1), self.order[X])
                col.c_chart[item] = value
            else:
                col.c_chart[item] = was + value

        else:
            # Items of the form phrase(I, X/[Y|Ys], K)
            item = (I, X, Ys)
            was = col.i_chart.get(item)
            if was is None:
                col.waiting_for[Ys[0]].add(item)
                col.i_chart[item] = value
            else:
                col.i_chart[item] = was + value

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
    def next_token_weights(self, cols):

        # Let's try a backward chaining algorithm
        #
        # q(0, s) += 1
        # q(J, Y) += phrase(I, X/[Y], J) * q(I, X)
        #
        # p(W) += q(I, X) * phrase(I, X/[W], J)  where len(J) * terminal(W).
        #

        q = {}
        q[0, self.cfg.S] = self.cfg.R.one

        def run(top):
            stack = [Node(top, None, zero)]

            while stack:
                node = stack[-1]   # 👀

                # place neighbors above the node on the stack
                (J, Y) = node.node

                t = node.cursor

                if node.edges is None:
                    node.edges = [x for x in cols[J].waiting_for[Y] if len(x[2]) == 1]

                # cursor is at the end, all neighbors are done
                elif t == len(node.edges):
                    # clear the node from the stack
                    stack.pop()
                    # promote the incomplete value node.value to a complete value (q)
                    q[node.node] = node.value

                else:
                    (I, X, Ys) = arc = node.edges[t]
                    neighbor = (I, X)
                    neighbor_value = q.get(neighbor)
                    if neighbor_value is None:
                        stack.append(Node(neighbor, None, zero))
                    else:
                        # neighbor value is ready, advance the cursor, add the
                        # neighbors contribution to the nodes value
                        node.cursor += 1
                        node.value += cols[J].i_chart[arc] * neighbor_value

            return q[top]

        zero = self.cfg.R.zero

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        p = self.cfg.R.chart()
        col = cols[-1]
        for item, arc_weight in col.i_chart.items():
            (I, X, Ys) = item
            if len(Ys) == 1 and self.cfg.is_terminal(Ys[0]):
                node = (I, X)
                value = q.get(node)
                if value is None:
                    value = run(node)
                p[Ys[0]] += arc_weight * value

        return p


class Node:
    __slots__ = ('value', 'node', 'edges', 'cursor')
    def __init__(self, node, edges, value):
        self.node = node
        self.edges = edges
        self.value = value
        self.cursor = 0
