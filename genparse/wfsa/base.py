from arsenal import Integerizer
from collections import defaultdict
from functools import cached_property
from graphviz import Digraph


EPSILON = "ε"


def _lehmann(R, N, W):
    "Lehmann's (1977) algorithm."

    V = W.copy()
    U = W.copy()

    for j in N:
        V, U = U, V
        V = R.chart()
        s = R.star(U[j, j])
        for i in N:
            for k in N:
                # i ➙ j ⇝ j ➙ k
                V[i, k] = U[i, k] + U[i, j] * s * U[j, k]

    # add paths of length zero
    for i in N:
        V[i, i] += R.one

    return V


class WFSA:
    """
    Weighted finite-state automata
    """
    def __init__(self, R):
        self.R = R
        self.alphabet = set()
        self.states = set()
        self.delta = defaultdict(lambda: defaultdict(R.chart))
        self.start = R.chart()
        self.stop = R.chart()

    def __repr__(self):
        return f'{__class__.__name__}({self.dim} states)'

    def __str__(self):
        output = []
        output.append('{')
        for p in self.states:
            output.append(f'  {p} \t\t({self.start[p]}, {self.stop[p]})')
            for a, q, w in self.arcs(p):
                output.append(f'    {a}: {q}\t[{w}]')
        output.append('}')
        return "\n".join(output)

    @property
    def dim(self):
        return len(self.states)

    def add_state(self, q):
        self.states.add(q)

    def add_arc(self, i, a, j, w):
        self.add_state(i)
        self.add_state(j)
        self.alphabet.add(a)
        self.delta[i][a][j] += w

    def add_I(self, q, w):
        self.add_state(q)
        self.start[q] += w

    def add_F(self, q, w):
        self.add_state(q)
        self.stop[q] += w

    @property
    def I(self):
        for q, w in self.start.items():
            if w != self.R.zero:
                yield q, w

    @property
    def F(self):
        for q, w in self.stop.items():
            if w != self.R.zero:
                yield q, w

    def arcs(self, i=None, a=None):
        if i is not None:
            if a is not None:
                for j, w in self.delta[i][a].items():
                    if w != self.R.zero:
                        yield j, w
            else:
                for a, T in self.delta[i].items():
                    for j, w in T.items():
                        if w != self.R.zero:
                            yield a, j, w
        else:
            if a is None:
                for i in self.delta:
                    for a, T in self.delta[i].items():
                        for j, w in T.items():
                            if w != self.R.zero:
                                yield i, a, j, w
            else:
                raise NotImplementedError

    def rename(self, f):
        "Note: If `f` is not bijective, states may merge."
        m = self.spawn()
        for i, w in self.I:
            m.add_I(f(i), w)
        for i, w in self.F:
            m.add_F(f(i), w)
        for i, a, j, w in self.arcs():
            m.add_arc(f(i), a, f(j), w)
        return m

    def rename_apart(self, other):
        f = Integerizer()
        return (self.rename(lambda i: f((0,i))), other.rename(lambda i: f((1,i))))

    @cached_property
    def renumber(self):
        return self.rename(Integerizer())

    def spawn(self, *, keep_init=False, keep_arcs=False, keep_stop=False):
        "Returns a new WFSA in the same semiring."
        m = self.__class__(self.R)
        if keep_init:
            for q, w in self.I:
                m.add_I(q, w)
        if keep_arcs:
            for i,a,j,w in self.arcs():
                m.add_arc(i, a, j, w)
        if keep_stop:
            for q, w in self.F:
                m.add_F(q, w)
        return m

    def __call__(self, xs):
        self = self.epsremove
        prev = self.start
        for x in xs:
            curr = self.R.chart()
            for i in prev:
                for j, w in self.arcs(i, x):
                    curr[j] += prev[i] * w
            prev = curr
        total = self.R.zero
        for j, w in self.F:
            total += prev[j] * w
        return total

    @cached_property
    def epsremove(self):
        E = self.R.chart()
        for i, a, j, w in self.arcs():
            if a == EPSILON:
                E[i, j] += w
        S = _lehmann(self.R, self.states, E)
        new = self.spawn(keep_stop=True)
        for i, w_i in self.I:
            for k in self.states:
                new.add_I(k, w_i * S[i, k])
        for i, a, j, w_ij in self.arcs():
            if a == EPSILON: continue
            for k in self.states:
                new.add_arc(i, a, k, w_ij * S[j, k])
        return new

    @cached_property
    def K(self):
        W = self.R.chart()
        for i, a, j, w in self.arcs():
            W[i, j] += w
        return _lehmann(self.R, self.states, W)

    @cached_property
    def reverse(self):
        "creates a reversed machine"
        # create the new machine
        R = self.spawn()
        # reverse each arc
        for i, a, j, w in self.arcs():
            R.add_arc(j, a, i, w)
        # reverse initial and final states
        for q, w in self.F:
            R.add_I(q, w)
        for q, w in self.I:
            R.add_F(q, w)
        return R

    def __add__(self, other):
        self, other = self.rename_apart(other)
        U = self.spawn(keep_init=True, keep_arcs=True, keep_stop=True)
        # add arcs, initial and final states from argument
        for i, a, j, w in other.arcs():
            U.add_arc(i, a, j, w)
        for q, w in other.I:
            U.add_I(q, w)
        for q, w in other.F:
            U.add_F(q, w)
        return U

    def __mul__(self, other):
        if not isinstance(other, self.__class__): return other.__rmul__(self)

        self, other = self.rename_apart(other)
        C = self.spawn(keep_init=True, keep_arcs=True)
        # add arcs, initial and final states from argument
        for i, a, j, w in other.arcs():
            C.add_arc(i, a, j, w)
        for q, w in other.F:
            C.add_F(q, w)
        # connect the final states from `self` to initial states from `other`
        for i1, w1 in self.F:
            for i2, w2 in other.I:
                C.add_arc(i1, EPSILON, i2, w1 * w2)
        return C

    @property
    def zero(self):
        return self.__class__(self.R)

    @property
    def one(self):
        return self.__class__.lift(EPSILON, self.R.one)

    def star(self):
        return self.one + self.kleene_plus()

    def kleene_plus(self):
        "self^+"
        m = self.spawn(keep_init=True, keep_arcs=True, keep_stop=True)
        for i, w1 in self.F:
            for j, w2 in self.I:
                m.add_arc(i, EPSILON, j, w1*w2)
        return m

    def _repr_svg_(self):
        return self.graphviz()._repr_image_svg_xml()

    def graphviz(self, fmt=str, fmt_node=lambda x: ' '):
        g = Digraph(
            graph_attr=dict(rankdir='LR'),
            node_attr=dict(
                fontname='Monospace',
                fontsize='10',
                height='.05', width='.05',
                #margin="0.055,0.042"
                margin="0,0"
            ),
            edge_attr=dict(
                #arrowhead='vee',
                arrowsize='0.3',
                fontname='Monospace',
                fontsize='9'
            ),
        )
        f = Integerizer()
        for i,w in self.I:
            start = f'<start_{i}>'
            g.node(start, label='', shape='point', height='0', width='0')
            g.edge(start, str(f(i)), label=f'{fmt(w)}')
        for i in self.states:
            g.node(str(f(i)), label=str(fmt_node(i)), shape='circle')
        for i,w in self.F:
            stop = f'<stop_{i}>'
            g.node(stop, label='', shape='point', height='0', width='0')
            g.edge(str(f(i)), stop, label=f'{fmt(w)}')
        #for i, a, j, w in sorted(self.arcs()):
        for i, a, j, w in self.arcs():
            g.edge(str(f(i)), str(f(j)), label=f'{a}/{fmt(w)}')
        return g

    @classmethod
    def lift(cls, x, w, R=None):
        if R is None: R = w.__class__
        m = cls(R=R)
        m.add_I(0, R.one)
        m.add_arc(0, x, 1, w)
        m.add_F(1, R.one)
        return m

    @classmethod
    def from_string(cls, xs, R, w=None):
        m = cls(R)
        m.add_I(xs[:0], R.one)
        for i in range(len(xs)):
            m.add_arc(xs[:i], xs[i], xs[:i+1], R.one)
        m.add_F(xs, (R.one if w is None else w))
        return m

    @classmethod
    def from_strings(cls, Xs, R):
        m = cls(R)
        for xs in Xs:
            m.add_I(xs[:0])
            for i in range(len(xs)):
                m.add_arc(xs[:i], xs[i], xs[:i+1], R.one)
            m.add_F(xs)
        return m

    def total_weight(self):
        b = self.backward()
        return sum(self.start[i] * b[i] for i in self.start)

    def forward(self):
        K = self.K
        initial = self.R.chart()
        for i,w in self.I:
            initial[i] += w
        forward = self.R.chart()
        for i,j in K:
            forward[j] += initial[i] * K[i,j]
        return forward

    def backward(self):
        K = self.K
        final = self.R.chart()
        for i,w in self.F:
            final[i] += w
        backward = self.R.chart()
        for i,j in K:
            backward[i] += K[i,j] * final[j]
        return backward

    @cached_property
    def trim(self):

        forward = self.forward()
        backward = self.backward()

        # determine the set of active state, (i.e., those with nonzero forward and backward weights)
        active = {
            i
            for i in self.states
            if forward[i] != self.R.zero and backward[i] != self.R.zero
        }

        new = self.spawn()
        for i in active:
            new.add_I(i, self.start[i])
            new.add_F(i, self.stop[i])

        for i,a,j,w in self.arcs():
            if i in active and j in active:
                new.add_arc(i,a,j,w)

        return new
