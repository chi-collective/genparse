from collections import defaultdict
from .util import format_table


class Chart(dict):

    def __init__(self, semiring, vals=()):
        self.semiring = semiring
        super().__init__(vals)

    def __missing__(self, k):
        return self.semiring.zero

    def spawn(self):
        return Chart(self.semiring)

    def __add__(self, other):
        new = self.spawn()
        for k, v in self.items():
            new[k] += v
        for k, v in other.items():
            new[k] += v
        return new

    def product(self, ks):
        v = self.semiring.one
        for k in ks:
            v *= self[k]
        return v

    def copy(self):
        return self.spawn() + self

    def trim(self):
        return Chart(self.semiring, {k: v for k, v in self._items() if v != self.semiring.zero})

    def metric(self, other):
        assert isinstance(other, Chart)
        err = 0
        for x in self.keys() | other.keys():
            err = max(err, self.semiring.metric(self[x], other[x]))
        return err

    def _repr_html_(self):
        return ('<div style="font-family: Monospace;">'
                + format_table(self.trim().items(), headings=['key', 'value'])
                + '</div>')

    def __repr__(self):
        return repr({k: v for k, v in self.items() if v != self.semiring.zero})

    def assert_equal(self, want, *, domain=None, tol=1e-5, verbose=False, throw=True):
        if domain is None: domain = self.keys() | want.keys()
        assert verbose or throw
        for x in domain:
            if self.semiring.metric(self[x], want[x]) <= tol:
                if verbose:
                    print(colors.mark(True), x, self[x])
            else:
                if verbose:
                    print(colors.mark(False), x, self[x], want[x])
                if throw:
                    raise AssertionError(f'{x}: {self[x]} {want[x]}')
