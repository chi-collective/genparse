"""
Algorithms for solving left-linear or right-linear systems of equations over closed semirings.
"""
import html
from arsenal import Integerizer
from collections import defaultdict
from functools import lru_cache, cached_property
from graphviz import Digraph


class WeightedGraph:

    def __init__(self, WeightType):
        self.N = set()
        self.incoming = defaultdict(set)
        self.WeightType = WeightType
        self.E = WeightType.chart()

    def __iter__(self):
        return iter(self.E)

    def __getitem__(self, item):
        i,j = item
        return self.E[i,j]

    def __setitem__(self, item, value):
        i,j = item
        self.N.add(i)
        self.N.add(j)
        self.E[i,j] = value
        self.incoming[j].add(i)
        return self

    def closure(self):
        return WeightedGraph(self.WeightType, self.closure_scc_based())

    def closure_reference(self):
        return self._closure(self.E, self.N)

    def closure_scc_based(self):
        K = self.WeightType.chart()
        for i in self.N:
            b = self.WeightType.chart()
            b[i] = self.WeightType.one
            sol = self.linsolve(b)
            for j in sol:
                K[i,j] = sol[j]
        return K

    def linsolve(self, b):
        """
        Solve `x = x A + b` using block, upper-triangular decomposition.
        """
        sol = self.WeightType.chart()
        for block, B in self.Blocks:

            # Compute the total weight of entering the block at each entry j in the block
            enter = self.WeightType.chart()
            for j in block:
                enter[j] += b[j]
                for i in self.incoming[j]:
                    enter[j] += sol[i] * self.E[i,j]

            # Now, compute the total weight of completing the block
            for j,k in B:
                sol[k] += enter[j] * B[j,k]

        return sol

    def _closure(self, A, N):
        """
        Compute the reflexive, transitive closure of `A` for the block of nodes `N`.
        """
        A = self.E
        old = A.copy()
        # transitive closure
        for j in N:
            new = self.WeightType.chart()
            sjj = self.WeightType.star(old[j,j])
            for i in N:
                for k in N:
                    new[i,k] = old[i,k] + old[i,j] * sjj * old[j,k]
            old, new = new, old   # swap to repurpose space
        # reflexive closure
        for i in N: old[i,i] += self.WeightType.one
        return old

    def blocks(self, roots=None):
        "Return the directed acyclic graph of strongly connected components."
        return tarjan(self.incoming.__getitem__, roots if roots else self.N)

    @cached_property
    def Blocks(self, **kwargs):
        return [(block, self._closure(self.E, block)) for block in self.blocks(**kwargs)]

    def _repr_svg_(self):
        return self.graphviz()._repr_image_svg_xml()

    def graphviz(self, label_format=str, escape=lambda x: html.escape(str(x))):

        name = Integerizer()

        g = Digraph(
            node_attr=dict(
                fontname='Monospace', fontsize='9', height='0', width='0',
                margin="0.055,0.042", penwidth='0.15', shape='box', style='rounded',
            ),
            edge_attr=dict(
                penwidth='0.5', arrowhead='vee', arrowsize='0.5',
                fontname='Monospace', fontsize='8'
            ),
        )

        for i,j in self.E:
            if self.E[i,j] == self.WeightType.zero: continue
            g.edge(str(name(i)), str(name(j)), label=label_format(self.E[i,j]))

        for i in self.N:
            g.node(str(name(i)), label=escape(i))

        return g


def tarjan(successors, roots):
    """
    Tarjan's linear-time algorithm O(E + V) for finding the maximal
    strongly connected components.
    """

    # 'Low Link Value' of a node is the smallest id reachable by DFS, including itself.
    # low link values are initialized to each node's id.
    lowest = {}      # node -> position of the root of the SCC

    stack = []      # stack
    trail = set()   # set of nodes on the stack
    t = 0

    def dfs(v):
        # DFS pushes nodes onto the stack
        nonlocal t
        t += 1
        num = t
        lowest[v] = t
        trail.add(v)
        stack.append(v)

        for w in successors(v):
            if lowest.get(w) is None:
                # As usual, only recurse when we haven't already visited this node
                yield from dfs(w)
                # The recursive call will find a cycle if there is one.
                # `lowest` is used to propagate the position of the earliest
                # node on the cycle in the DFS.
                lowest[v] = min(lowest[v], lowest[w])
            elif w in trail:
                # Collapsing cycles.  If `w` comes before `v` in dfs and `w` is
                # on the stack, then we've detected a cycle and we can start
                # collapsing values in the SCC.  It might not be the maximal
                # SCC. The min and stack will take care of that.
                lowest[v] = min(lowest[v], lowest[w])

        if lowest[v] == num:
            # `v` is the root of an SCC; We're totally done with that subgraph.
            # nodes above `v` on the stack are an SCC.
            C = []
            while True:   # pop until we reach v
                w = stack.pop()
                trail.remove(w)
                C.append(w)
                if w == v: break
            yield frozenset(C)

    for v in roots:
        if lowest.get(v) is None:
            yield from dfs(v)
