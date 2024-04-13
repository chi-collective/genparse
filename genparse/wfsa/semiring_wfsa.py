"""
Implementation of semiring-weighted finite-state automata.
"""

from functools import cached_property
from frozendict import frozendict

from genparse.wfsa import base
EPSILON = base.EPSILON


class WFSA(base.WFSA):

    @cached_property
    def minimize(self):
        """
        Implements Brzozowski's_minimization algorithm.
        See https://ralphs16.github.io/src/CatLectures/HW_Brzozowski.pdf and
        https://link.springer.com/chapter/10.1007/978-3-642-39274-0_17.
        """
        return self.reverse.determinize.trim.reverse.determinize.trim

    @cached_property
    def determinize(self):
        """
        Mohri (2009)'s "on-the-fly" determinization semi-algorithm.
        https://link.springer.com/chapter/10.1007/978-3-642-01492-5_6

        Use with caution as this method may not terminate.
        """

        self = self.epsremove.push

        def _powerarcs(Q):

            U = {a: self.R.chart() for a in self.alphabet}

            for i, u in Q.items():
                for a, j, v in self.arcs(i):
                    U[a][j] += u * v

            for a in U:
                R = U[a]
                W = sum(R.values(), start=self.R.zero)

                if 0:
                    # If we cannot extract a common factor, then all of the arcs will have weight one
                    yield a, frozendict(R), self.R.one

                else:
                    yield a, frozendict({p: W**(-1) * R[p] for p in R}), W

        D = self.spawn()

        stack = []
        visited = set()

        Q = frozendict({i: w for i, w in self.I})
        D.add_I(Q, self.R.one)
        stack.append(Q)
        visited.add(Q)

        while stack:
            P = stack.pop()
            for a, Q, w in _powerarcs(P):
                if Q not in visited:
                    stack.append(Q)
                    visited.add(Q)
                D.add_arc(P, a, Q, w)

        for Q in D.states:
            for q in Q:
                D.add_F(Q, Q[q] * self.stop[q])

        return D

    @cached_property
    def push(self):
        "Weight pushing algorithm (Mohri, 2001)."
        V = self.backward
        new = self.spawn()
        for i in self.states:
            if V[i] == self.R.zero: continue
            new.add_I(i, self.start[i] * V[i])
            new.add_F(i, V[i]**(-1) * self.stop[i])
            for a, j, w in self.arcs(i):
                new.add_arc(i, a, j, V[i]**(-1) * w * V[j])
        return new
