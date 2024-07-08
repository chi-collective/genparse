"""
Code For Grammar Aligned Decoding (Park et al., 2024)
"""

import html
import numpy as np
from arsenal import Integerizer, colors
from arsenal.maths import sample

from graphviz import Digraph
from genparse import Float, EOS
import re

Z = lambda x: 1 if sum(x.values()) == 0 else sum(x.values())  # avoid division by zero


class Sampler:
    """
    This class is used to navigate trough the sample trie, and update its node.
    It also computes the approximate expected future grammaticality.
    """

    def __init__(self, lm1, lm2):
        self.lm1 = lm1  # This is supposed to be the LLM
        self.lm2 = lm2  # This is supposed to be the CFG
        self.S = set()
        self.root = Node(1.0, None, None)
        assert lm1.V == lm2.V
        self.V = lm1.V

    def sample(self):
        """Samples a string and at the same time adds the string to the sample trie"""
        curr = self.root
        a = ''
        while a != EOS:
            for sym in self.V:
                self.next_node(curr, sym)  # ensure that every child is initialized
            cs = list(curr.children)

            # print(f" lm1 : {[(b,self.lm1.p_next(curr.prefix)[b]) for b in cs]}")
            # print(f" lm2 : {[(b,self.lm2.p_next(curr.prefix)[b]) for b in cs]}")

            ms = [
                curr.children[b].mass
                * self.lm1.p_next(curr.prefix)[b]
                / Z(self.lm1.p_next(curr.prefix))
                * self.lm2.p_next(curr.prefix)[b]
                / Z(self.lm2.p_next(curr.prefix))
                for b in cs
            ]

            a = cs[sample(ms)]
            curr = curr.children[a]
            print(curr.mass, curr.prefix)
            print(cs, ms)
        # self.update_backward(curr) # backward update of the efg values.
        curr.update_backward(self.lm1, self.lm2)
        return curr.prefix

    def next_node(self, node, a):
        """Given a node and a symbol, returns the next node.
        If the next node does not exist, it creates the node and assign mass to it.
        This method should be used together with the GAD sampler"""
        if node.children and a in node.children.keys():
            return node.children[a]
        # elif a==EOS:
        #     next = Node(self.lm2.model(node.prefix + a), node, prefix=node.prefix + a) # Base case: EOS
        #     node.children[a] = next
        #     return next
        elif a == EOS:
            next = Node(Float.one, node, prefix=node.prefix + a)
            node.children[a] = next
            return next
        else:
            next = Node(
                self.lm1.p_next(node.prefix)[a]
                / Z(self.lm1.p_next(node.prefix))
                * self.lm2.p_next(node.prefix)[a]
                / Z(self.lm2.p_next(node.prefix)),
                node,
                prefix=node.prefix + a,
            )
            node.children[a] = next
            return next


class Node:
    __slots__ = ('mass', 'parent', 'children', 'prefix', '_mass')

    def __init__(
        self, mass, parent, children=None, prefix=None
    ):  # Clemente: Now children is initialized to empty set
        self.mass = mass
        self.parent = parent
        self.children = {} if children is None else children
        self.prefix = '' if prefix is None else prefix
        self._mass = mass  # bookkeeping: remember the original mass

    @classmethod
    def build_trie_cfg(cls, S, cfglm):
        Root = cls(1, None, prefix='')
        for stryng in S:
            curr = Root
            for i in range(len(stryng)):
                a = stryng[i]
                if curr.children and a in curr.children.keys():
                    curr = curr.children[a]
                else:
                    next = cls(cfglm.p_next(stryng[0:i])[a], curr, prefix=curr.prefix + a)
                    curr.children[a] = next
                    curr = next
        return Root

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

    def update_backward(self, lm1, lm2):
        """This method updates backwards the mass of the node -- leaf to root ---,
        following the approximate EFG scheme of Park et al.(2024)
        """
        if self.parent is None:
            return
        parent = self.parent

        parent.mass = sum(
            [
                lm1.p_next(parent.prefix)[a]
                / Z(lm1.p_next(parent.prefix))
                * lm2.p_next(parent.prefix)[a]
                / Z(lm2.p_next(parent.prefix))
                * node.mass
                for a, node in parent.children.items()
            ]
        )  # WE have to normalize since we need the next-token weight

        parent.update_backward(lm1, lm2)

    # def update(self):
    #     "Restore the invariant that self.mass = sum children mass."
    #     if self.children is not None:
    #         self.mass = sum(y.mass for y in self.children.values())
    #     if self.parent is not None:
    #         self.parent.update()

    def graphviz(
        self,
        fmt_edge=lambda x, a, y: f'{html.escape(str(a))}/{y.mass/x.mass:.2g}',
        # fmt_node=lambda x: ' ',
        fmt_node=lambda x: (
            # f'{x.mass}/{x._mass:.2g}' if x.mass > 0 else f'{x._mass:.2g}'
            f'{x.mass:.2g}'
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
                g.node(str(f(x)), label=str(fmt_node(x)) + '/' + x.prefix, shape='box')
            else:
                g.node(
                    str(f(x)),
                    label=str(fmt_node(x)) + '/' + x.prefix,
                    shape='box',
                    fillcolor='gray',
                )
        return g
