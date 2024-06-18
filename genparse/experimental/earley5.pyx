#!python
#cython: initializedcheck=False
#cython: boundscheck=False
###cython: wraparound=False
###cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: cdivision=True
#cython: overflowcheck=False
#cython: language_level=3
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

import numpy as np
from collections import defaultdict
from functools import lru_cache

# TODO: move heap into its own file and make it importable from arsenal, but for
# now its pasted at the bottom of this file
#
#from arsenal.datastructures.pdict import pdict
#from arsenal.datastructures.heap cimport LocatorMaxHeap   # TODO: should be able to cimport this as it is written in cython

import numpy as np
from arsenal import Integerizer, colors

from genparse.cfglm import EOS, add_EOS
from genparse.linear import WeightedGraph
from genparse.lm import LM
from genparse.semiring import Boolean

from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from libcpp.utility cimport pair


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


ctypedef (long,long,long) IncompleteItem
#ctypedef (long,long) CompleteItem
ctypedef (long,long) CompleteItem

#cdef struct IncompleteItem:
#    int I
#    int X
#    int Ys

#cdef struct CompleteItem:
#    int I
#    int X


cdef class Column:

    cdef readonly:
        long k
        dict i_chart
        dict c_chart
        LocatorMaxHeap Q
        object waiting_for
#    cdef public:
#        unordered_map[int, unordered_set[IncompleteItem]] _waiting_for
#    cdef public:
#        unordered_map[CompleteItem, double] _c_chart

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
        dict _chart
        set V
        int eos
        Column _initial_column
        dict R_outgoing
        dict rhs
        int ORDER_MAX
        object intern_Ys
        object _encode_symbol

        long[:] order
        long[:] first_Ys
        long[:] rest_Ys
        long[:] unit_Ys

    def __init__(self, cfg):

        cfg = cfg.nullaryremove(binarize=True).unarycycleremove().renumber()
        self.cfg = cfg

        # cache of chart columns
        self._chart = {}

        # Topological ordering on the grammar symbols so that we process unary
        # rules in a topological order.
        order = cfg._unary_graph_transpose().buckets
        self.order = np.zeros(len(self.cfg.N), dtype=int)
        for x in self.cfg.N:
            self.order[x] = order[x]

        self.ORDER_MAX = max(self.order)

        # left-corner graph
        R = WeightedGraph(Boolean)
        for r in cfg:
            if len(r.body) == 0:
                continue
            A = r.head
            B = r.body[0]
            R[A, B] += Boolean.one
        self.R_outgoing = dict(R.outgoing)

        intern_Ys = Integerizer()
        assert intern_Ys(()) == 0

        for r in self.cfg:
            for p in range(len(r.body) + 1):
                intern_Ys.add(r.body[p:])

        self.rhs = {}
        for X in self.cfg.N:
            self.rhs[X] = []
            for r in self.cfg.rhs[X]:
                if r.body == (): continue
                self.rhs[X].append((r.w, intern_Ys(r.body)))

        self.intern_Ys = intern_Ys
        self._encode_symbol = Integerizer()
        for x in self.cfg.N:
            self._encode_symbol(x)

        self.first_Ys = np.zeros(len(intern_Ys), dtype=int)
        self.rest_Ys = np.zeros(len(intern_Ys), dtype=int)
        self.unit_Ys = np.zeros(len(intern_Ys), dtype=int)

        cdef int code
        for Ys, code in list(self.intern_Ys.items()):
            self.unit_Ys[code] = (len(Ys) == 1)
            if len(Ys) > 0:
                Y = Ys[0]
                if self.cfg.is_terminal(Y):
                    self.first_Ys[code] = self._encode_symbol(Y)
                else:
                    self.first_Ys[code] = Y
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

    def next_column(self, list[Column] prev_cols, str token):
        cdef long I, J, X, Ys
        cdef IncompleteItem item
        cdef double value, y

        prev_col = prev_cols[-1]
        next_col = Column(prev_cols[-1].k + 1)
        next_col_c_chart = next_col.c_chart
        prev_col_i_chart = prev_col.i_chart

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        for item in prev_col.waiting_for[self._encode_symbol(token)]:
            (I, X, Ys) = item
            value = prev_col_i_chart[item]
            self._update(next_col, I, X, self.rest_Ys[Ys], value)

        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
        Q = next_col.Q
        while Q:
            (J, Y) = node = Q.pop()[0]
            col_J = prev_cols[J]
            col_J_i_chart = col_J.i_chart
            y = next_col_c_chart[node]
            for item in col_J.waiting_for[Y]:
                (I, X, Ys) = item
                value = col_J_i_chart[item] * y
                self._update(next_col, I, X, self.rest_Ys[Ys], value)

        self.PREDICT(next_col)

        return next_col

    cdef void PREDICT(self, Column prev_col):
        cdef IncompleteItem item
        cdef dict rhs
        cdef long Ys, k   # XXX: X and Y might be a string | integer.
        # PREDICT: phrase(K, X/Ys, K) += rule(X -> Ys) with some filtering heuristics
        k = prev_col.k
        prev_col_chart = prev_col.i_chart

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

        rhs = self.rhs
        for X in reachable:
            for w, Ys in rhs.get(X, ()):
                item = (k, X, Ys)
                was = prev_col_chart.get(item)
                if was is None:
                    Y = self.first_Ys[Ys]
                    prev_col.waiting_for[Y].add(item)
                    prev_col_chart[item] = w
                else:
                    prev_col_chart[item] = was + w

    cdef inline void _update(self, Column col, long I, long X, long Ys, double value):
        cdef CompleteItem c_item
        cdef IncompleteItem i_item
        K = col.k
        if Ys == 0:
            # Items of the form phrase(I, X/[], K)
            c_item = (I, X)
            if col.c_chart.get(c_item) is None:
                col.Q[c_item] = -((K - I) * self.ORDER_MAX + self.order[X])
                col.c_chart[c_item] = value
            else:
                col.c_chart[c_item] += value

        else:
            # Items of the form phrase(I, X/[Y|Ys], K)
            i_item = (I, X, Ys)
            if col.i_chart.get(i_item) is None:
                col.waiting_for[self.first_Ys[Ys]].add(i_item)
                col.i_chart[i_item] = value
            else:
                col.i_chart[i_item] += value

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
        cdef int I, X, Ys
        q = {}
        q[0, self.cfg.S] = 1

        is_terminal = self.cfg.is_terminal

        col = cols[-1]
        col_i_chart = col.i_chart

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * word(J, Y, K)
        p = {}

        for Y in col.waiting_for:
            W = self._encode_symbol[Y]
            if is_terminal(W):    # XXX: expensive to lookup the symbol like this
                total = 0.0
                for (I, X, Ys) in col.waiting_for[Y]:
                    if self.unit_Ys[Ys]:
                        node = (I, X)
                        value = q.get(node)
                        if value is None:
                            value = self._helper(node, cols, q)
                        total += col_i_chart[I, X, Ys] * value
                p[W] = total
        return self.cfg.R.chart(p)

    cdef double _helper(self, (int, int) top, list[Column] cols, dict q):
        cdef list[Node] stack
        cdef Node node
        cdef int I, J, X, Y
        cdef IncompleteItem x

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


"""
Heap data structures with optional
 - Locators
 - Top-k (Bounded heap)

"""
import numpy as np

Vt = np.double
cdef double NaN = np.nan

# TODO: Use the C++ standard library's implementation of a vector of doubles.
cdef class Vector:

    cdef public:
        int cap
        int end
        double[:] val

    def __init__(self, cap):
        self.cap = cap
        self.val = np.zeros(self.cap, dtype=Vt)
        self.end = 0

    cpdef int push(self, double x):
        i = self.end
        self.ensure_size(i)
        self.val[i] = x
        self.end += 1
        return i

    cpdef object pop(self):
        "pop from the end"
        assert 0 < self.end
        self.end -= 1
        v = self.val[self.end]
        self.val[self.end] = NaN
        return v

    cdef void grow(self):
        self.cap *= 2
        new = np.empty(self.cap, dtype=Vt)
        new[:self.end] = self.val[:self.end]
        self.val = new

    cdef void ensure_size(self, int i):
        "grow in needed"
        if self.val.shape[0] < i + 1: self.grow()

    def __getitem__(self, int i):
        assert i < self.end
        return self.get(i)

    cdef double get(self, int i):
        return self.val[i]

    def __setitem__(self, int i, double v):
        assert i < self.end
        self.set(i, v)

    cdef void set(self, int i, double v):
        self.val[i] = v

    def __len__(self):
        return self.end

    def __repr__(self):
        return repr(self.val[:self.end])


cdef class MaxHeap:

    cdef public:
        Vector val

    def __init__(self, cap=2**8):
        self.val = Vector(cap)
        self.val.push(np.nan)

    def __len__(self):
        return len(self.val) - 1   # subtract one for dummy root element

    def pop(self):
        v = self.peek()
        self._remove(1)
        return v

    def peek(self):
        return self.val.val[1]

    def push(self, v):
        # put new element last and bubble up
        return self.up(self.val.push(v))

    cdef void swap(self, int i, int j):
        assert i < self.val.end
        assert j < self.val.end
        self.val.val[i], self.val.val[j] = self.val.val[j], self.val.val[i]

    cdef int up(self, int i):
        while 1 < i:
            p = i // 2
            if self.val.val[p] < self.val.val[i]:
                self.swap(i, p)
                i = p
            else:
                break
        return i

    cdef int down(self, int i):
        n = self.val.end
        while 2*i < n:
            a = 2 * i
            b = 2 * i + 1
            c = i
            if self.val.val[c] < self.val.val[a]:
                c = a
            if b < n and self.val.val[c] < self.val.val[b]:
                c = b
            if c == i:
                break
            self.swap(i, c)
            i = c
        return i

    def _update(self, int i, double old, double new):
        assert i < self.val.end
        if old == new: return i   # value unchanged
        self.val.val[i] = new         # perform change
        if old < new:             # increased
            return self.up(i)
        else:                     # decreased
            return self.down(i)

    def _remove(self, int i):
        # update the locator stuff for last -> i
        last = self.val.end - 1
        self.swap(i, last)
        old = self.val.pop()
        # special handling for when the heap has size one.
        if i == last: return
        self._update(i, old, self.val.val[i])

    def check(self):
        # heap property
        for i in range(2, self.val.end):
            assert self.val[i] <= self.val[i // 2], (self.val[i // 2], self.val[i])   # child <= parent


cdef class LocatorMaxHeap(MaxHeap):
    """
    Dynamic heap. Maintains max of a map, via incrementally maintained partial
    aggregation tree. Also known a priority queue with 'locators'.

    This data structure efficiently maintains maximum of the priorities of a set
    of keys. Priorites may increase or decrease. (Many max-heap implementations
    only allow increasing priority.)

    """

    cdef public:
        dict key
        dict loc

    def __init__(self, **kw):
        super().__init__(**kw)
        self.key = {}   # map from index `i` to `key`
        self.loc = {}   # map from `key` to index in `val`

    def __repr__(self):
        return repr({k: self[k] for k in self.loc})

    def pop(self):
        k,v = self.peek()
        super().pop()
        return k,v

    def popitem(self):
        return self.pop()

    def peek(self):
        return self.key[1], super().peek()

    def _remove(self, int i):
        # update the locator stuff for last -> i
        last = self.val.end - 1
        self.swap(i, last)
        old = self.val.pop()
        # remove the key/loc/val associated with the deleted node.
        self.loc.pop(self.key.pop(last))
        # special handling for when the heap has size one.
        if i == last: return
        self._update(i, old, self.val.val[i])

    def __delitem__(self, k):
        self._remove(self.loc[k])

    def __contains__(self, k):
        return k in self.loc

    def __getitem__(self, k):
        return self.val.val[self.loc[k]]

    def __setitem__(self, k, v):
        "upsert (update or insert) value associated with key."
        cdef int i
        if k in self:
            # update
            i = self.loc[k]
            super()._update(i, self.val[i], v)
        else:
            # insert (put new element last and bubble up)
            i = self.val.push(v)
            # Annoyingly, we have to write key/loc here the super class's push
            # method doesn't allow us to intervene before the up call.
            self.val[i] = v
            self.loc[k] = i
            self.key[i] = k
            # fix invariants
            self.up(i)

    cdef void swap(self, int i, int j):
        assert i < self.val.end
        assert j < self.val.end
        self.val.val[i], self.val.val[j] = self.val.val[j], self.val.val[i]

        self.key[i], self.key[j] = self.key[j], self.key[i]
        self.loc[self.key[i]] = i
        self.loc[self.key[j]] = j

    def check(self):
        super().check()
        for key in self.loc:
            assert self.key[self.loc[key]] == key
        for i in range(1, self.val.end):
            assert self.loc[self.key[i]] == i


class MinMaxHeap:

    def __init__(self, **kw):
        self.max = LocatorMaxHeap(**kw)
        self.min = LocatorMaxHeap(**kw)   # will pass negative values here

    def __contains__(self, k):
        return k in self.max

    def __setitem__(self, k, v):
        self.max[k] = v
        self.min[k] = -v

    def peekmin(self):
        k, v = self.min.peek()
        return k, -v

    def peekmax(self):
        return self.max.peek()

    def popmax(self):
        k, v = self.max.pop()
        self.min._remove(self.min.loc[k])  # remove it from the min heap
        return k, v

    def popmin(self):
        k, v = self.min.pop()
        self.max._remove(self.max.loc[k])  # remove it from the min heap
        return k, -v

    def check(self):
        self.min.check()
        self.max.check()

    def __len__(self):
        return len(self.max)

    def __repr__(self):
        return repr(self.max)

    def map(self):
        return {k: self.max[k] for k in self.max.loc}


class BoundedMaxHeap(MinMaxHeap):

    def __init__(self, maxsize, **kw):
        super().__init__(**kw)
        self.maxsize = maxsize

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if len(self) > self.maxsize:
            if v < self.peekmin()[1]:  # smaller than the smallest element.
                return
            else:
                self.popmin()   # evict the smallest element

    def pop(self):
        return self.popmax()

    def check(self):
        super().check()
        assert len(self.max) <= self.maxsize
        assert len(self.min) <= self.maxsize
