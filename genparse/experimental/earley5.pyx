#!python
#cython: initializedcheck=False
#cython: boundscheck=False
###cython: wraparound=False
###cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: cdivision=True
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

import numpy as np
from collections import defaultdict
from functools import lru_cache

#from arsenal.datastructures.pdict import pdict
from arsenal.datastructures.heap import LocatorMaxHeap   # TODO: should be able to cimport this as it is written in cython

from genparse.cfglm import EOS, add_EOS
from genparse.linear import WeightedGraph
from genparse.lm import LM
from genparse.semiring import Boolean

from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from cpython.object cimport PyObject


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

    def clear_cache(self):
        self.model.clear_cache()

cdef struct IncompleteItem:
    int I
    int X
    int Ys


cdef class Column:

    cdef readonly:
        int k
        dict i_chart
        dict c_chart
        object Q
        object waiting_for
#    cdef public:
#        unordered_map[int, unordered_set[IncompleteItem]] _waiting_for

    def __init__(self, k):
        self.k = k
        self.i_chart = {}
        self.c_chart = {}

        # Within column J, this datastructure maps nonterminals Y to a set of items
        #   Y => {(I, X, Ys) | phrase(I,X/[Y],J) ≠ 0}
        self.waiting_for = defaultdict(set)

        # priority queue used when first filling the column
        self.Q = LocatorMaxHeap()


cdef class Earley:
    """
    Implements a semiring-weighted version Earley's algorithm that runs in O(N^3|G|) time.
    Warning: Assumes that nullary rules and unary chain cycles have been removed
    """

    cdef:
        object cfg
        dict order
        dict _chart
        set V
        int eos
        Column _initial_column
        dict R_outgoing
        int ORDER_MAX
        object intern_Ys

        object[:] first_Ys    # terminals are currently string
        long[:] rest_Ys
        long[:] unit_Ys

    def __init__(self, cfg):

        cfg = cfg.nullaryremove(binarize=True).unarycycleremove().renumber()
        self.cfg = cfg

        # cache of chart columns
        self._chart = {}

        # Topological ordering on the grammar symbols so that we process unary
        # rules in a topological order.
        self.order = cfg._unary_graph_transpose().buckets

        self.ORDER_MAX = max(self.order.values())

        # left-corner graph
        R = WeightedGraph(Boolean)
        for r in cfg:
            if len(r.body) == 0:
                continue
            A = r.head
            B = r.body[0]
            R[A, B] += Boolean.one
        self.R_outgoing = dict(R.outgoing)

        import numpy as np
        from arsenal import Integerizer, colors
        intern_Ys = Integerizer()
        assert intern_Ys(()) == 0

        for r in self.cfg:
            for p in range(len(r.body) + 1):
                intern_Ys.add(r.body[p:])

        self.intern_Ys = intern_Ys

        self.first_Ys = np.zeros(len(intern_Ys), dtype=object)
        self.rest_Ys = np.zeros(len(intern_Ys), dtype=int)
        self.unit_Ys = np.zeros(len(intern_Ys), dtype=int)

        cdef int code
        for Ys, code in list(self.intern_Ys.items()):
            self.unit_Ys[code] = (len(Ys) == 1)
            if len(Ys) > 0:
                self.first_Ys[code] = Ys[0]
                self.rest_Ys[code] = intern_Ys(Ys[1:])

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
        return value if value is not None else 0

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

    def next_column(self, list[Column] prev_cols, token):
        cdef int I, J, X, Ys
        cdef (int, int, int) item

        prev_col = prev_cols[-1]
        next_col = Column(prev_cols[-1].k + 1)
        next_col_c_chart = next_col.c_chart
        prev_col_i_chart = prev_col.i_chart

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        for item in prev_col.waiting_for[token]:
            (I, X, Ys) = item
            self._update(next_col, I, X, self.rest_Ys[Ys], prev_col_i_chart[item])

        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
        Q = next_col.Q
        while Q:
            (J, Y) = node = Q.pop()[0]
            col_J = prev_cols[J]
            col_J_i_chart = col_J.i_chart
            y = next_col_c_chart[node]
            for item in col_J.waiting_for[Y]:
                (I, X, Ys) = item
                self._update(next_col, I, X, self.rest_Ys[Ys], col_J_i_chart[item] * y)

        self.PREDICT(next_col)

        return next_col

    cdef PREDICT(self, Column prev_col):
        cdef int Ys, k   # XXX: X and Y might be a string | integer.
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
            for Y in self.R_outgoing.get(X, ()):
                if Y not in reachable:
                    reachable.add(Y)
                    agenda.append(Y)

        rhs = self.cfg.rhs
        for X in reachable:
            for r in rhs[X]:
                Ys = self.intern_Ys(r.body)
                if Ys == 0:    # 0 is the empty body
                    continue
                item = (k, X, Ys)
                was = prev_col_chart.get(item)
                if was is None:
                    Y = self.first_Ys[Ys]
                    prev_col_waiting_for[Y].add(item)
                    prev_col_chart[item] = r.w
                else:
                    prev_col_chart[item] = was + r.w

    cdef _update(self, Column col, int I, int X, int Ys, double value):
        K = col.k
        if Ys == 0:
            # Items of the form phrase(I, X/[], K)
            item = (I, X)
            was = col.c_chart.get(item)
            if was is None:
                col.Q[item] = -((K - I) * self.ORDER_MAX + self.order[X])
                col.c_chart[item] = value
            else:
                col.c_chart[item] = was + value

        else:
            # Items of the form phrase(I, X/[Y|Ys], K)
            item = (I, X, Ys)
            was = col.i_chart.get(item)
            if was is None:
                col.waiting_for[self.first_Ys[Ys]].add(item)
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
    # nonterminal ordering.  Thus, we can efficiently evaluate these equations
    # by backward chaining.
    #
    # The final output is the vector
    #
    # p(W) += q(I, X) * phrase(I, X/[W], J)  where len(J) * terminal(W).
    #
    cpdef object next_token_weights(self, list[Column] cols):
        cdef double total
        q = {}
        q[0, self.cfg.S] = 1

        is_terminal = self.cfg.is_terminal

        col = cols[-1]
        col_waiting_for = col.waiting_for
        col_i_chart = col.i_chart

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        p = {}

        for Y in col_waiting_for:
            if is_terminal(Y):
                total = 0.0
                for (I, X, Ys) in col_waiting_for[Y]:
                    if self.unit_Ys[Ys]:
                        node = (I, X)
                        value = q.get(node)
                        if value is None:
                            value = self._helper(node, cols, q)
                        total += col_i_chart[I, X, Ys] * value
                p[Y] = total
        return self.cfg.R.chart(p)

    cdef double _helper(self, (int, int) top, list[Column] cols, dict q):
        cdef list[Node] stack
        cdef Node node
        cdef int I, J, X, Y
        stack = [Node(top, None, 0.0)]

        while stack:
            node = stack[-1]   # 👀

            # place neighbors above the node on the stack
            (J, Y) = node.node

            t = node.cursor

            if node.edges is None:
                node.edges = [x for x in cols[J].waiting_for[Y] if self.unit_Ys[x[2]]]

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
                    stack.append(Node(neighbor, None, 0.0))
                else:
                    # neighbor value is ready, advance the cursor, add the
                    # neighbors contribution to the nodes value
                    node.cursor += 1
                    node.value += cols[J].i_chart[arc] * neighbor_value

        return q[top]


cdef class Node:
    cdef public:
        double value
        (int, int) node
        list edges
        int cursor
    def __init__(self, node, edges, value):
        self.node = node
        self.edges = edges
        self.value = value
        self.cursor = 0
