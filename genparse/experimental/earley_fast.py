import numpy as np
from arsenal import Integerizer

from collections import defaultdict


from genparse.cfglm import EOS, add_EOS, CFG
from genparse.lm import LM
from genparse.semiring import Boolean, Float

from genpa_rs import Earley as _Earley, EarleyBool as _EarleyBool


class BoolCFGLM(LM):
    def __init__(self, cfg, alg='earley'):
        if EOS not in cfg.V:
            cfg = add_EOS(cfg)
        if cfg.R != Boolean:
            cfg = cfg.map_values(lambda x: Boolean(x > 0), Boolean)
        assert alg == 'earley', 'only support fast Earley'
        self.model = Earley(cfg.prefix_grammar)
        super().__init__(eos=EOS, V=cfg.V)

    def p_next(self, context):
        assert set(context) <= self.V, f'OOVs detected: {set(context) - self.V}'
        p = self.model.next_token_weights(context)
        p = Boolean.chart(p).trim()
        return Float.chart({w: 1 for w in p})

    def __call__(self, context):
        return float(super().__call__(context) > 0)

    def clear_cache(self):
        self.model.clear_cache()

    @classmethod
    def from_string(cls, x, semiring=Boolean, **kwargs):
        return cls(CFG.from_string(x, semiring), **kwargs)


class EarleyLM(LM):
    def __init__(self, cfg):
        if EOS not in cfg.V:
            cfg = add_EOS(cfg)
        self.cfg = cfg
        self.model = Earley(cfg.prefix_grammar)
        super().__init__(V=cfg.V, eos=EOS)

    def p_next(self, context):
        assert set(context) <= self.V, f'OOVs detected: {set(context) - self.V}'
        return Float.chart(self.model.next_token_weights(context)).normalize()

    def clear_cache(self):
        self.model.clear_cache()


def deep_convert_to_bool(obj):
    if isinstance(obj, Boolean):
        return obj.score
    elif isinstance(obj, list):
        return [deep_convert_to_bool(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_convert_to_bool(item) for item in obj)
    elif isinstance(obj, set):
        return {deep_convert_to_bool(item) for item in obj}
    elif isinstance(obj, dict):
        return {key: deep_convert_to_bool(value) for key, value in obj.items()}
    else:
        return obj


class Earley:
    """
    Wrapper of the Rust Earley parser.

    Original documentation:

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

        empty_weight = sum(
            (r.w for r in self.cfg.rhs[self.cfg.S] if r.body == ()), cfg.R.zero
        )
        outgoing = {}
        for k, v in self.R_outgoing.items():
            outgoing[k] = list(v)

        if self.cfg.R == Boolean:
            print('using boolean')
            self.impl = _EarleyBool(
                deep_convert_to_bool(self.rhs),
                self.cfg.S,
                self.order,
                self.ORDER_MAX,
                outgoing,
                self.first_Ys,
                self.rest_Ys,
                self.unit_Ys,
                self.cfg.V,
                empty_weight.score,
            )
        else:
            self.impl = _Earley(
                self.rhs,
                self.cfg.S,
                self.order,
                self.ORDER_MAX,
                outgoing,
                self.first_Ys,
                self.rest_Ys,
                self.unit_Ys,
                self.cfg.V,
                empty_weight,
            )

    def __call__(self, x):
        return self.impl.compute_weight(tuple(x))

    def next_token_weights(self, x):
        return self.impl.p_next(tuple(x))

    def clear_cache(self):
        self.impl.clear_cache()
