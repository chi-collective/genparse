"""
Tracing
"""

import html

from arsenal import Integerizer, colors
from arsenal.maths import sample
from graphviz import Digraph

from genparse.semiring import Float
from genparse.util import format_table


class generation_tree:
    def __init__(self, lm, **opts):
        tracer = TraceSWOR()
        D = Float.chart()
        while tracer.root.mass > 0:
            with tracer:
                s, p = lm.sample(draw=tracer, **opts)
                D[s] += p
        D = Float.chart((k, D[k]) for k in sorted(D))
        self.D = D
        self.tracer = tracer

    def _repr_html_(self):
        return format_table([[self.D, self.tracer]])


class Tracer:
    """
    This class lazily materializes the probability tree of a generative process by program tracing.
    """

    def __init__(self):
        self.root = Node(1.0, None, None)
        self.cur = None

    def __call__(self, p, context=None):
        "Sample an action while updating the trace cursor and tree data structure."

        if not isinstance(p, dict):
            p = dict(enumerate(p))

        cur = self.cur

        if cur.children is None:  # initialize the newly discovered node
            cur.children = {a: Node(cur.mass * p[a], parent=cur) for a in p if p[a] > 0}
            self.cur.context = (
                context  # store the context, which helps detect trace divergence
            )

        if context != cur.context:
            print(colors.light.red % 'ERROR: trace divergence detected:')
            print(colors.light.red % 'trace context:', self.cur.context)
            print(colors.light.red % 'calling context:', context)
            raise ValueError((p, cur))

        a = cur.sample()
        self.cur = cur.children[a]  # advance the cursor
        return a


class Node:
    __slots__ = ('mass', 'parent', 'children', 'context', '_mass')

    def __init__(self, mass, parent, children=None, context=None):
        self.mass = mass
        self.parent = parent
        self.children = children
        self.context = context
        self._mass = mass  # bookkeeping: remember the original mass

    def sample(self):
        cs = list(self.children)
        ms = [c.mass for c in self.children.values()]
        return cs[sample(ms)]

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
        "Restore the invariant that self.mass = sum children mass."
        if self.children is not None:
            self.mass = sum(y.mass for y in self.children.values())
        if self.parent is not None:
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
            if x.children is None:
                continue
            for a, y in x.children.items():
                g.edge(str(f(x)), str(f(y)), label=f'{fmt_edge(x,a,y)}')
                q.append(y)
        for x in xs:
            if x.children is not None:
                g.node(str(f(x)), label=str(fmt_node(x)), shape='box')
            else:
                g.node(str(f(x)), label=str(fmt_node(x)), shape='box', fillcolor='gray')
        return g


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
