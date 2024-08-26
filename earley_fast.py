from pprint import pprint

import numpy as np
from arsenal import Integerizer

from collections import defaultdict

from arsenal.datastructures.heap import LocatorMaxHeap
from mpmath import isint

from genparse.cfglm import EOS, add_EOS, locally_normalize, CFG
from genparse.lm import LM
from genparse import Float

from genpa_rs import Earley as _Earley


class Earley:
    """
    Implements a semiring-weighted version Earley's algorithm that runs in $\mathcal{O}(N^3|G|)$ time.
    Note that nullary rules and unary chain cycles will be been removed, altering the
    set of derivation trees.
    """

    def __init__(self, cfg):
        cfg = cfg.nullaryremove(binarize=True).unarycycleremove().renumber()
        self.cfg = cfg

        # cache of chart columns
        self._chart = {}

        # Topological ordering on the grammar symbols so that we process unary
        # rules in a topological order.
        self.order = cfg._unary_graph_transpose().buckets

        self.ORDER_MAX = 1 + max(self.order.values())

        # left-corner graph
        R_outgoing = defaultdict(set)
        for r in cfg:
            if len(r.body) == 0:
                continue
            A = r.head
            B = r.body[0]
            if cfg.is_terminal(B):
                continue
            R_outgoing[A].add(B)
        self.R_outgoing = R_outgoing

        # Integerize rule right-hand side states
        intern_Ys = Integerizer()
        assert intern_Ys(()) == 0

        for r in self.cfg:
            for p in range(len(r.body) + 1):
                intern_Ys.add(r.body[p:])

        self.intern_Ys = intern_Ys

        self.rhs = {}
        for X in self.cfg.N:
            assert isinstance(X, int)
            self.rhs[X] = []
            for r in self.cfg.rhs[X]:
                if r.body == ():
                    continue
                self.rhs[X].append((r.w, intern_Ys(r.body)))
        print('\n\n\nrhs', self.rhs)

        self.first_Ys = np.zeros(len(intern_Ys), dtype=object)
        self.rest_Ys = np.zeros(len(intern_Ys), dtype=int)
        self.unit_Ys = np.zeros(len(intern_Ys), dtype=bool)

        for Ys, code in list(self.intern_Ys.items()):
            self.unit_Ys[code] = len(Ys) == 1
            if len(Ys) > 0:
                self.first_Ys[code] = Ys[0]
                self.rest_Ys[code] = intern_Ys(Ys[1:])

        self.first_Ys = self.first_Ys.tolist()
        self.rest_Ys = self.rest_Ys.tolist()
        self.unit_Ys = self.unit_Ys.tolist()

        empty_weight = sum(r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ())
        outgoing = {}
        for k, v in self.R_outgoing.items():
            outgoing[k] = list(v)

        self.impl = _Earley(
            self.rhs,
            self.cfg.S,
            self.order,
            self.ORDER_MAX,
            outgoing,
            self.first_Ys,
            self.rest_Ys,
            self.unit_Ys,
            empty_weight,
        )

    def __call__(self, x):
        return self.impl.compute_weight(tuple(x))

    def clear_cache(self):
        self.impl.clear_cache()
