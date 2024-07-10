"""
Tracing
"""

import html

from arsenal import Integerizer, colors
from arsenal.maths import sample
from graphviz import Digraph

from genparse.semiring import Float
from genparse.util import format_table
from genparse.lm import LazyProb

import numpy as np


class generation_tree:
    def __init__(self, lm, remaining_mass=0.0, **opts):
        tracer = TraceSWOR()
        D = Float.chart()
        while tracer.root.mass > remaining_mass:
            with tracer:
                s, p = lm.sample(draw=tracer, **opts)
                D[s] += p
        D = Float.chart((k, D[k]) for k in sorted(D))
        self.D = D
        self.tracer = tracer

    def _repr_html_(self):
        return format_table([[self.D, self.tracer]])

    def __repr__(self):
        return str(self.D) + str(self.tracer)


def separate_keys_vals(x):
    if isinstance(x, LazyProb):
        return x.keys(), x.values()
    elif isinstance(x, np.ndarray):
        return range(len(x)), x
    else:
        return list(x.keys()), np.array(list(x.values()))


class Tracer:
    """
    This class lazily materializes the probability tree of a generative process by program tracing.
    """

    def __init__(self):
        self.root = Node(idx=-1, mass=1.0, parent=None)
        self.cur = None

    def __call__(self, p, context=None):
        "Sample an action while updating the trace cursor and tree data structure."

        keys, p = separate_keys_vals(p)
        cur = self.cur

        if cur.child_masses is None:
            cur.child_masses = cur.mass * p
            cur.context = context

        if context != cur.context:
            print(colors.light.red % 'ERROR: trace divergence detected:')
            print(colors.light.red % 'trace context:', self.cur.context)
            print(colors.light.red % 'calling context:', context)
            raise ValueError((p, cur))

        a = cur.sample()
        if a not in cur.active_children:
            cur.active_children[a] = Node(
                idx=a,
                mass=cur.child_masses[a],
                parent=cur,
                token=keys[a],
            )
        self.cur = cur.active_children[a]
        return keys[a]


class Node:
    __slots__ = (
        'idx',
        'mass',
        'parent',
        'token',
        'child_masses',
        'active_children',
        'context',
        '_mass',
    )

    def __init__(
        self,
        idx,
        mass,
        parent,
        token=None,
        child_masses=None,
        context=None,
    ):
        self.idx = idx
        self.mass = mass
        self.parent = parent
        self.token = token  # used for visualization
        self.child_masses = child_masses
        self.active_children = {}
        self.context = context
        self._mass = mass  # bookkeeping: remember the original mass

    def sample(self):
        return sample(self.child_masses)

    def p_next(self):
        return Float.chart((a, c.mass / self.mass) for a, c in self.children.items())

    # TODO: untested
    def sample_path(self):
        curr = self
        path = []
        P = 1
        while True:
            p = curr.p_next()
            a = curr.sample()
            P *= p[a]
            curr = curr.children[a]
            if not curr.children:
                break
            path.append(a)
        return (P, path, curr)

    def update(self):
        # TODO: Fennwick tree alternative, sumheap
        # TODO: optimize this by subtracting from masses, instead of resumming
        "Restore the invariant that self.mass = sum children mass."
        if self.parent is not None:
            self.parent.child_masses[self.idx] = self.mass
            self.parent.mass = np.sum(self.parent.child_masses)
            self.parent.update()

    def graphviz(
        self,
        fmt_edge=lambda x, a, y: f'{html.escape(str(a))}/{y._mass/x._mass:.2g}',
        # fmt_node=lambda x: ' ',
        fmt_node=lambda x: (
            f'{x.mass}/{x._mass:.2g}' if x.mass > 0 else f'{x._mass:.2g}'
        ),
    ):
        "Create a graphviz instance for this subtree"
        g = Digraph(
            graph_attr=dict(rankdir='LR'),
            node_attr=dict(
                fontname='Monospace',
                fontsize='10',
                height='.05',
                width='.05',
                margin='0.055,0.042',
            ),
            edge_attr=dict(arrowsize='0.3', fontname='Monospace', fontsize='9'),
        )
        f = Integerizer()
        xs = set()
        q = [self]
        while q:
            x = q.pop()
            xs.add(x)
            if x.child_masses is None:
                continue
            for a, y in x.active_children.items():
                a = y.token if y.token is not None else a
                g.edge(str(f(x)), str(f(y)), label=f'{fmt_edge(x,a,y)}')
                q.append(y)
        for x in xs:
            if x.child_masses is not None:
                g.node(str(f(x)), label=str(fmt_node(x)), shape='box')
            else:
                g.node(str(f(x)), label=str(fmt_node(x)), shape='box', fillcolor='gray')
        return g

    def downstream_nodes(self):
        q = [self]
        while q:
            x = q.pop()
            yield x
            if x.child_masses is None:
                continue
            for a, y in x.active_children.items():
                q.append(y)


class TraceSWOR(Tracer):
    """
    Sampling without replacement 🤝 Program tracing.
    """

    def __enter__(self):
        self.cur = self.root

    def __exit__(self, *args):
        self.cur.mass = 0  # we will never sample this node again.
        self.cur.update()  # update invariants

    def _repr_svg_(self):
        return self.root.graphviz()._repr_image_svg_xml()

    def sixel_render(self):
        try:
            from sixel import converter
            import sys
            from io import BytesIO

            c = converter.SixelConverter(BytesIO(self.root.graphviz()._repr_image_png()))
            c.write(sys.stdout)
        except ImportError:
            import warnings

            warnings.warn('Install imgcat or sixel to enable rendering.')
            print(self)

    def __repr__(self):
        return self.root.graphviz().source
