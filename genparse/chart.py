from arsenal import colors

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
        return Chart(self.semiring, self)

    def trim(self):
        return Chart(self.semiring, {k: v for k, v in self.items() if v != self.semiring.zero})

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

    def __str__(self, style_value=lambda k,v: str(v)):

        def key(k):
            return -self.semiring.metric(self[k], self.semiring.zero)

        return (
            'Chart {\n' +
            '\n'.join(
                f'  {k!r}: {style_value(k, self[k])},' for k in sorted(self, key=key) if self[k] != self.semiring.zero
            )
            + '\n}'
        )

    def assert_equal(self, want, *, domain=None, tol=1e-5, verbose=False, throw=True):
        if not isinstance(want, Chart): want = self.semiring.chart(want)
        if domain is None: domain = self.keys() | want.keys()
        assert verbose or throw
        errors = []
        for x in domain:
            if self.semiring.metric(self[x], want[x]) <= tol:
                if verbose: print(colors.mark(True), x, self[x])
            else:
                if verbose: print(colors.mark(False), x, self[x], want[x])
                errors.append(x)
        if throw:
            for x in errors:
                raise AssertionError(f'{x}: {self[x]} {want[x]}')

    def argmax(self):
        return max(self, key=self.__getitem__)

    def argmin(self):
        return min(self, key=self.__getitem__)

    def max(self):
        return max(self.values())

    def min(self):
        return min(self.values())

    def sum(self):
        return sum(self.values())

    def normalize(self):
        Z = self.sum()
        return self.semiring.chart((x, v/Z) for x, v in self.items())

    def project(self, f):
        "Apply the function `f` to each key; summing when f-transformed keys overlap."
        out = self.semiring.chart()
        for k, v in self.items():
            out[f(k)] += v
        return out
