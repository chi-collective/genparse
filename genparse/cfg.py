import re
import nltk
import numpy as np
import graphviz

from collections import defaultdict, Counter
from functools import cached_property
from itertools import product

from .chart import Chart
from .semiring import Semiring, Boolean
from .util import colors, format_table
from .linear import WeightedGraph


def _gen_nt(prefix=''):
    _gen_nt.i += 1
    return f'{prefix}@{_gen_nt.i}'
_gen_nt.i = 0


# This is just the data structure with none of the methods that make it a weighted language
class FSA:
    def __init__(self, start, edges, stop):
        self.start = dict(start)
        self.edges = edges
        self.stop = dict(stop)

        self.states = set()
        self.states.update(s for s in self.start)
        self.states.update(s for s in self.stop)
        self.states.update(s for s, _, _, _ in self.edges)
        self.states.update(s for _, _, s, _ in self.edges)

        self.alphabet = {a for _, a, _, _ in self.edges}

    def arcs(self):
        return self.edges

    @classmethod
    def from_string(cls, xs, semiring):
        "Make a straight-line automaton from the string `xs`."
        return FSA(
            [(0, semiring.one)],
            [(i, xs[i], i+1, semiring.one) for i in range(len(xs))],
            [(len(xs), semiring.one)],
        )


class Slash:

    def __init__(self, Y, Z, id):
        self.Y, self.Z = Y, Z
        self._hash = hash((Y, Z, id))
        self.id = id

    def __repr__(self):
        if self.id == 0:
            return f'{self.Y}/{self.Z}'
        else:
            return f'{self.Y}/{self.Z}@{self.id}'

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, Slash)
            and self.Y == other.Y
            and self.Z == other.Z
            and self.id == other.id
        )


class Rule:

    def __init__(self, w, head, body):
        self.w = w
        self.head = head
        self.body = body
        self._hash = hash((head, body))

    def __iter__(self):
        return iter((self.head, self.body))

    def __eq__(self, other):
        return (isinstance(other, Rule)
                and self.w == other.w
                and self._hash == other._hash
                and other.head == self.head
                and other.body == self.body)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f'{self.w}: {self.head} → {" ".join(map(str, self.body))}'


class Derivation:

    def __init__(self, r, x, *ys):
        assert isinstance(r, Rule) or r is None
        self.r = r
        self.x = x
        self.ys = ys

    # Warning: Currently, Derivations compare equal even if they have different rules.
    def __hash__(self):
#        return hash((self.r, self.x, self.ys))
        return hash((self.x, self.ys))

    def __eq__(self, other):
#        return (self.r, self.x, self.ys) == (other.r, other.x, other.ys)
        return isinstance(other, Derivation) and (self.x, self.ys) == (other.x, other.ys)

    def __repr__(self):
        open = colors.dark.white % '('
        close = colors.dark.white % ')'
        children = ' '.join(str(y) for y in self.ys)
        return f'{open}{self.x} {children}{close}'

    def weight(self):
        "Compute this weight this `Derivation`."
        W = self.r.w
        for y in self.ys:
            if isinstance(y, Derivation):
                W *= y.weight()
        return W

    def Yield(self):
        if isinstance(self, Derivation):
            return tuple(w for y in self.ys for w in Derivation.Yield(y))
        else:
            return (self,)

    def to_nltk(self):
        if not isinstance(self, Derivation): return self
        return nltk.Tree(str(self.x), [Derivation.to_nltk(y) for y in self.ys])

    def _repr_html_(self):
#        return f'<div style="text-align: center;"><span style="color: magenta;">{self.weight()}</span></br>{self.to_nltk()._repr_svg_()}</div>'
        return self.to_nltk()._repr_svg_()


class CFG:

    def __init__(self, R: 'semiring', S: 'start symbol', V: 'terminal vocabulary'):
        self.R = R      # semiring
        self.V = V      # alphabet
        self.N = {S}    # nonterminals
        self.S = S      # unique start symbol
        self.rules = [] # rules

    def __repr__(self):
        return "\n".join(f"{p}" for p in self)

    def _repr_html_(self):
        return f'<pre style="width: fit-content; text-align: left; border: thin solid black; padding: 0.5em;">{self}</pre>'

    @classmethod
    def from_string(cls, string, semiring, comment="#", start='S', is_terminal=lambda x: not x[0].isupper()):
        V = set()
        cfg = cls(R=semiring, S=start, V=V)
        string = string.replace('->', '→')   # synonym for the arrow
        for line in string.split('\n'):
            line = line.strip()
            if not line or line.startswith(comment): continue
            try:
                [(w, lhs, rhs)] = re.findall('(.*):\s*(\S+)\s*→\s*(.*)$', line)
                lhs = lhs.strip()
                rhs = rhs.strip().split()
                for x in rhs:
                    if is_terminal(x):
                        V.add(x)
                cfg.add(semiring.from_string(w), lhs, *rhs)
            except ValueError as e:
                raise ValueError(f'bad input line:\n{line}')
        return cfg

    def __getitem__(self, X):
        "Return the CFG for language of the nonterminal `X`."
        assert self.is_nonterminal(X)
        new = self.spawn(S=X)
        for r in self:
            new.add(r.w, r.head, *r.body)
        return new

    def __call__(self, input):
        "Compute the total weight of the `input` sequence."
        return self._parse_chart(input)[0,self.S,len(input)]

    def _parse_chart(self, input):
        "Implements CKY algorithm for evaluating the total weight of the `input` sequence."
        if not self.in_cnf(): self = self.cnf
        (nullary, terminal, binary) = self._cnf
        N = len(input)
        # nullary rule
        c = self.R.chart()
        for i in range(N+1):
            c[i,self.S,i] += nullary
        # preterminal rules
        for i in range(N):
            for r in terminal[input[i]]:
                c[i,r.head,i+1] += r.w
        # binary rules
        for span in range(1, N + 1):
            for i in range(N - span + 1):
                k = i + span
                for j in range(i + 1, k):
                    for r in binary:
                        X, [Y, Z] = r.head, r.body
                        c[i,X,k] +=  r.w * c[i,Y,j] * c[j,Z,k]
        return c

    def language(self, depth):
        "Enumerate strings generated by this cfg by derivations up to a the given `depth`."
        lang = self.R.chart()
        for d in self.derivations(self.S, depth):
            lang[d.Yield()] += d.weight()
        return lang

    @cached_property
    def rhs(self):
        rhs = defaultdict(list)
        for r in self:
            rhs[r.head].append(r)
        return rhs

    def is_terminal(self, x):
        return x in self.V

    def is_nonterminal(self, X):
        return not self.is_terminal(X)

    def __iter__(self):
        return iter(self.rules)

    @property
    def size(self):
        return sum(1 + len(r.body) for r in self)

    @property
    def num_rules(self):
        return len(self.rules)

    def spawn(self, *, R=None, S=None, V=None):
        return self.__class__(R=self.R if R is None else R,
                              S=self.S if S is None else S,
                              V=set(self.V) if V is None else V)

    def add(self, w, head, *body):
        if w == self.R.zero: return   # skip rules with weight zero
        self.N.add(head)
        r = Rule(w, head, body)
        self.rules.append(r)
        return r

    def rename(self, f):
        new = self.spawn(S = f(self.S))
        for r in self:
            new.add(r.w, f(r.head), *((y if self.is_terminal(y) else f(y)
                                       for y in r.body)))
        return new

    def assert_equal(self, other, verbose=False, throw=True):
        assert verbose or throw
        if isinstance(other, str): other = self.__class__.from_string(other, self.R)
        if verbose:
            # TODO: need to check the weights in the print out; we do it in the assertion
            S = set(self.rules)
            G = set(other.rules)
            for r in sorted(S | G, key=str):
                if r in S and r in G: continue
                #if r in S and r not in G: continue
                #if r not in S and r in G: continue
                print(
                    colors.mark(r in S),
                    #colors.mark(r in S and r in G),
                    colors.mark(r in G),
                    r,
                )
        assert not throw or Counter(self.rules) == Counter(other.rules), \
            f'\n\nhave=\n{str(self)}\nwant=\n{str(other)}'

    def treesum(self, **kwargs):
        return self.agenda()[self.S]

    def trim(self, bottomup_only=False):

        C = set(self.V)
        C.update(e.head for e in self.rules if len(e.body) == 0)

        incoming = defaultdict(list)
        outgoing = defaultdict(list)
        for e in self:
            incoming[e.head].append(e)
            for b in e.body:
                outgoing[b].append(e)

        agenda = set(C)
        while agenda:
            x = agenda.pop()
            for e in outgoing[x]:
                if all((b in C) for b in e.body):
                    if e.head not in C:
                        C.add(e.head)
                        agenda.add(e.head)

        if bottomup_only: return self._trim(C)

        T = {self.S}
        agenda.update(T)
        while agenda:
            x = agenda.pop()
            for e in incoming[x]:
                #assert e.head in T
                for b in e.body:
                    if b not in T and b in C:
                        T.add(b)
                        agenda.add(b)

        return self._trim(T)

    def cotrim(self):
        return self.trim(bottomup_only=True)

    def _trim(self, symbols):
        new = self.spawn()
        for p in self:
            if p.head in symbols and p.w != self.R.zero and set(p.body) <= symbols:
                new.add(p.w, p.head, *p.body)
        return new

    #___________________________________________________________________________
    # Derivation enumeration

    def derivations(self, X, H):
        "Enumerate derivations of symbol X with height <= H"
        if X is None: X = self.S

        if self.is_terminal(X):
            yield X

        elif H <= 0:
            return

        else:
            for r in self.rhs[X]:
                for ys in self._derivations_list(r.body, H-1):
                    yield Derivation(r, X, *ys)

    def _derivations_list(self, X, H):
        if len(X) == 0:
            yield ()
        else:
            for x in self.derivations(X[0], H):
                for xs in self._derivations_list(X[1:], H):
                    yield (x, *xs)

    def derivations_of(self, s):
        "Enumeration of derivations with yield `s`"

        def p(X,I,K):
            if self.is_terminal(X):
                if K-I == 1 and s[I] == X:
                    yield X
                else:
                    return
            else:
                for r in self.rhs[X]:
                    for ys in ps(r.body, I, K):
                        yield Derivation(r, X, *ys)

        def ps(X,I,K):
            if len(X) == 0:
                if K-I == 0:
                    yield ()
            else:
                for J in range(I, K+1):
                    for x in p(X[0], I, J):
                        for xs in ps(X[1:], J, K):
                            yield (x, *xs)

        return p(self.S, 0, len(s))

    #___________________________________________________________________________
    # Transformations

    def unaryremove(self):
        """
        Return an equivalent grammar with no unary rules.
        """

        # compute the matrix closure of the unary rules, so we can unfold them
        # into the preterminal and binary rules.
        A = WeightedGraph(self.R)
        for r in self:
            if len(r.body) == 1 and self.is_nonterminal(r.body[0]):
                A[r.head, r.body[0]] += r.w

        A.N |= self.N

        W = A.closure_scc_based()

        new = self.spawn()
        for r in self:
            if len(r.body) == 1 and self.is_nonterminal(r.body[0]): continue
            for Y in self.N:
                new.add(W[Y, r.head]*r.w, Y, *r.body)

        return new

    def nullaryremove(self, binarize=True, **kwargs):
        """
        Return an equivalent grammar with no nullary rules except for one at the
        start symbol.
        """
        # A really wide rule can take a very long time because of the power set
        # in this rule so it is really important to binarize.
        if binarize: self = self.binarize()
        self = self.separate_start()
        return self._push_null_weights(self.null_weight(), **kwargs)

    def null_weight(self):
        """
        Compute the map from nonterminal to total weight of generating the
        empty string starting from that nonterminal.
        """
        ecfg = self.spawn(V=set())
        for p in self:
            if not any(self.is_terminal(y) for y in p.body):
                ecfg.add(p.w, p.head, *p.body)
        return ecfg.agenda()

    def null_weight_start(self):
        return self.null_weight()[self.S]

    def _push_null_weights(self, null_weight, recovery=False, rename=lambda x: f'${x}'):
        """Returns a grammar that generates the same weighted language but it is
        nullary-free at all nonterminals except its start symbol.  [Assumes that
        S does not appear on any RHS; call separate_start to ensure this.]

        The nonterminals with nonzero null_weight will be eliminated from the
        grammar.  They will be repaced with nullary-free variants that are
        marked according to `rename` (the default option is to mark them with a
        dollar sign prefix).

        Bonus (Hygiene property): Any nonterminal that survives the this
        transformation is guaranteed to generate the same weighted language.

        """

        # Warning: this method might have issues when `separate_start` hasn't
        # been run before.  So we run it rather than leaving it up to chance.
        assert self.S not in {y for r in self for y in r.body}

        def f(x):
            "Rename nonterminal if necessary"
            if null_weight[x] == self.R.zero or x == self.S:   # not necessary; keep old name
                return x
            else:
                return rename(x)

        rcfg = self.spawn()
        rcfg.add(null_weight[self.S], self.S)

        if recovery:
            for x in self.N:
                if f(x) == x: continue
                rcfg.add(null_weight[x], x)
                rcfg.add(self.R.one, x, f(x))

        for r in self:

            if len(r.body) == 0: continue  # drop nullary rule

            for B in product([0, 1], repeat=len(r.body)):
                v, new_body = r.w, []

                for i, b in enumerate(B):
                    if b:
                        v *= null_weight[r.body[i]]
                    else:
                        new_body.append(f(r.body[i]))

                # exclude the cases that would be new nullary rules!
                if len(new_body) > 0:
                    rcfg.add(v, f(r.head), *new_body)

        return rcfg

    def separate_start(self):
        "Ensure that the start symbol does not appear on the RHS of any rule."
        # create a new start symbol if the current one appears on the rhs of any existing rule
        if self.S in {y for r in self for y in r.body}:
            S = _gen_nt(self.S)
            new = self.spawn(S = S)
            # preterminal rules
            new.add(self.R.one, S, self.S)
            for r in self:
                new.add(r.w, r.head, *r.body)
            return new
        else:
            return self

    def separate_terminals(self):
        "Ensure that the each terminal is produced by a preterminal rule."
        one = self.R.one
        new = self.spawn()

        _preterminal = {}
        def preterminal(x):
            y = _preterminal.get(x)
            if y is None:
                y = new.add(one, _gen_nt(), x)
                _preterminal[x] = y
            return y

        for r in self:
            if len(r.body) == 1 and self.is_terminal(r.body[0]):
                new.add(r.w, r.head, *r.body)
            else:
                new.add(r.w, r.head, *((preterminal(y).head if self.is_terminal(y) else y) for y in r.body))

        return new

    def binarize(self):
        new = self.spawn()

        stack = list(self)
        while stack:
            p = stack.pop()
            if len(p.body) <= 2:
                new.add(p.w, p.head, *p.body)
            else:
                stack.extend(self._fold(p, [(0, 1)]))

        return new

    def _fold(self, p, I):

        # new productions
        P, heads = [], []
        for (i, j) in I:
            head = _gen_nt()
            heads.append(head)
            body = p.body[i:j+1]
            P.append(Rule(self.R.one, head, body))

        # new "head" production
        body = tuple()
        start = 0
        for (end, n), head in zip(I, heads):
            body += p.body[start:end] + (head,)
            start = n+1
        body += p.body[start:]
        P.append(Rule(p.w, p.head, body))

        return P

    @cached_property
    def cnf(self):
        new = self.separate_terminals().nullaryremove(binarize=True).trim().unaryremove().trim()
        assert new.in_cnf()
        return new

    # TODO: make CNF grammars a speciazed subclass of CFG.
    @cached_property
    def _cnf(self):
        nullary = self.R.zero
        terminal = defaultdict(list)
        binary = []
        for r in self:
            if len(r.body) == 0:
                nullary += r.w
                assert r.head == self.S
            elif len(r.body) == 1:
                terminal[r.body[0]].append(r)
                assert self.is_terminal(r.body[0])
            else:
                assert len(r.body) == 2
                binary.append(r)
                assert self.is_nonterminal(r.body[0])
                assert self.is_nonterminal(r.body[1])
        return (nullary, terminal, binary)

    def in_cnf(self):
        "Return true of the grammar is in CNF."
        for r in self:
            assert r.head in self.N
            if len(r.body) == 0 and r.head == self.S:
                continue
            elif len(r.body) == 1 and self.is_terminal(r.body[0]):
                continue
            elif len(r.body) == 2 and all(self.is_nonterminal(y) and y != self.S for y in r.body):
                continue
            else:
                return False
        return True

    def unfold(self, i, k):
        assert isinstance(i, int) and isinstance(k, int)
        s = self.rules[i]
        assert self.is_nonterminal(s.body[k])

        wp = self.R.zero
        new = self.spawn()
        for j, r in enumerate(self):
            if j != i:
                new.add(r.w, r.head, *r.body)

        for r in self.rhs[s.body[k]]:
            new.add(s.w*r.w, s.head, *s.body[:k], *r.body, *s.body[k+1:])

        return new

    def agenda(self, tol=1e-12):
        "Agenda-based semi-naive evaluation"
        old = self.R.chart()

        # precompute the mapping from updates to where they need to go
        routing = defaultdict(list)
        for r in self:
            for k in range(len(r.body)):
                routing[r.body[k]].append((r, k))

        # Dependency analysis to determine a reasonable prioritization order
        # 1) Form the dependency graph
        deps = WeightedGraph(Boolean)
        for r in self:
            for y in r.body:
                deps[r.head, y] += Boolean.one
        deps.N |= self.N; deps.N |= self.V
        # 2) Run the SCC analysis, extract its results
        blocks = list(deps.blocks())
        bucket = {y: i for i, block in enumerate(reversed(blocks)) for y in block}

        # helper function
        def update(x, W):
            change[bucket[x]][x] += W

        change = defaultdict(lambda: self.R.chart())
        for a in self.V:
            update(a, self.R.one)

        for r in self:
            if len(r.body) == 0:
                update(r.head, r.w)

        B = len(blocks)
        b = 0
        while b < B:

            if len(change[b]) == 0:
                b += 1
                continue

            u,v = change[b].popitem()

            new = old[u] + v

            if self.R.metric(old[u], new) <= tol: continue

            for r, k in routing[u]:

                W = r.w
                for j in range(len(r.body)):
                    if u == r.body[j]:
                        if j < k:    W *= new
                        elif j == k: W *= v
                        else:        W *= old[u]
                    else:
                        W *= old[r.body[j]]

                update(r.head, W)

            old[u] = new

        return Chart(self.R, old)

    def naive_bottom_up(self, *, tol=1e-12, timeout=100_000):

        def _approx_equal(U, V):
            return all((self.R.metric(U[X], V[X]) <= tol) for X in self.N)

        R = self.R
        V = R.chart()
        counter = 0
        while counter < timeout:
            U = self._bottom_up_step(V)
            if _approx_equal(U, V): break
            V = U
            counter += 1
        return Chart(self.R, V)

    def _bottom_up_step(self, V):
        R = self.R
        one = R.one
        U = R.chart()
        for a in self.V:
            U[a] = one
        for p in self:
            update = p.w
            for X in p.body:
                if self.is_nonterminal(X):
                    update *= V[X]
            U[p.head] += update
        return U

    def intersect(self, fsa):
        "Return a CFG that denoting the pointwise product of `self` and `fsa`."
        if isinstance(fsa, (str, list, tuple)): fsa = FSA.from_string(fsa, self.R)
        new_start = self.S
        new = self.spawn(S = new_start)
        for r in self:
            for qs in product(fsa.states, repeat=1+len(r.body)):
                new.add(r.w, (qs[0], r.head, qs[-1]), *((qs[i], r.body[i], qs[i+1]) for i in range(len(r.body))))
        for qi, wi in fsa.start.items():
            for qf, wf in fsa.stop.items():
                new.add(wi*wf, new_start, (qi, self.S, qf))
        for i, a, j, w in fsa.arcs():
            new.add(w, (i, a, j), a)
        return new

    def prefix_weight(self, input):
        return self.prefix_grammar(input)

    @cached_property
    def prefix_grammar(self):
        #if not self.in_cnf(): self = self.cnf
        return PrefixGrammar(self)

    def gensym(self, x):
        assert x not in self.V
        if x not in self.N: return x
        i = 1
        while f'{x}@{i}' in self.N:
            i += 1
        return f'{x}@{i}'

    def derivatives(self, s):
        "Return the sequence of derivatives for each prefix of `s`."
        M = len(s)
        D = [self]
        for m in range(M):
            D.append(D[m].derivative(s[m]))
        return D

    # Implementation note: This implementation of the derivative grammar
    # performs nullary elimination at the same time.
    def derivative(self, a, id=0):
        "Return a grammar that generates the derivative with respect to `a`."
        if isinstance(a, list): return self.derivatives(a)[-1]
        def slash(x, y): return Slash(x, y, id=id)
        D = self.spawn(S = slash(self.S, a))
        U = self.null_weight()
        for r in self:
            D.add(r.w, r.head, *r.body)
            delta = self.R.one
            for k, y in enumerate(r.body):
                if slash(r.head, a) in self.N: continue   # SKIP!
                if self.is_terminal(y):
                    if y == a:
                        D.add(delta*r.w, slash(r.head, a), *r.body[k+1:])
                else:
                    D.add(delta*r.w, slash(r.head, a), slash(r.body[k], a), *r.body[k+1:])
                delta *= U[y]
        return D#.trim()


# TODO: replace this code with the transduction version!
class PrefixGrammar(CFG):
    """
    Left-derivative transformation returns a grammar that computes all left
    derivatives when it is intersected with a straight-line automaton accepting
    a given input string.
    """

    def __init__(self, parent):

        self.parent = parent
        other = self._other
        free = self._free
        top = self._top
        super().__init__(
            S = top(parent.S),
            V = parent.V,
            R = parent.R,
        )

        # Our construction for `other` assumes that there are new empty strings
        # to get those back we add one more kind of item that unions them.
        #
        # TODO: can we merge `top` with `other`?
        for x in parent.N:
            self.add(self.R.one, top(x), free(x))
            self.add(self.R.one, top(x), other(x))

        # keep all of the original rules
        for r in parent:
            self.add(r.w, r.head, *r.body)

        # invisible suffix.  These are empty "future strings".  The rules add
        # 'free' rules with the exact same structure, but different base cases,
        # as they generate empty strings only
        for x in parent.V:
            self.add(self.R.one, free(x))        # generates the empty string
        for r in parent:
            self.add(r.w, free(r.head), *(free(z) for z in r.body))

        # The `other` items (better name pending) are possibly incomplete items
        # that all nonempty prefixes of their base nonterminal's language.  Top
        # is the same, but it includes the empty string.
        #
        # visible prefix - Below, we carefully move the `other`-cursor along
        # each rule body.. The `other` are such that they have an `other`-spine
        # that separates the /visible/ prefix from the /invisible/ suffix.
        for x in parent.V:
            self.add(self.R.one, other(x), x)    # generates the usual string
        for r in parent:
            for k in range(len(r.body)):
                self.add(r.w, other(r.head), *r.body[:k], other(r.body[k]), *(free(z) for z in r.body[k+1:]))

    def spawn(self, *, R=None, S=None, V=None):   # override or else we will spawn
        return CFG(R=self.R if R is None else R,
                   S=self.S if S is None else S,
                   V=set(self.V) if V is None else V)

    def _other(self, x):
        return self.parent.gensym(f'{x}⚡')

    def _free(self, x):
        return self.parent.gensym(f'{x}🔥')

    def _top(self, x):
        return self.parent.gensym(f'#{x}')
