from collections import defaultdict

from genparse.lm import LM
from genparse.cfglm import EOS, add_EOS, locally_normalize, CFG
from genparse.semiring import Float


class CKYLM(LM):
    """
    Probabilistic Context-Free Grammar Language Model.

    Uses CKY and the prefix grammar transformation for efficient inference.
    """

    def __init__(self, cfg, **kwargs):
        if EOS not in cfg.V:
            cfg = add_EOS(cfg)
        self.cfg = cfg
        self.pfg = self.cfg.cnf.prefix_grammar.cnf
        self.model = IncrementalCKY(self.pfg, **kwargs)
        super().__init__(V=cfg.V, eos=EOS)

    def p_next(self, context):
        assert set(context) <= self.V, f'OOVs detected: {set(context) - self.V}'
        return self.model.p_next(context).normalize()

    @classmethod
    def from_string(cls, x, semiring=Float, **kwargs):
        return cls(locally_normalize(CFG.from_string(x, semiring), **kwargs))

    def clear_cache(self):
        self.model.clear_cache()


class IncrementalCKY:
    def __init__(self, cfg):
        cfg = cfg.renumber()
        self.cfg = cfg
        self.S = cfg.S

        # cache columns of the chart indexed by prefix
        self._chart = {}

        [self.nullary, self.terminal, binary] = cfg._cnf
        r_y_xz = defaultdict(list)
        for r in binary:  # binary rules
            r_y_xz[r.body[0]].append(r)
        self.r_y_xz = r_y_xz

    def clear_cache(self):
        self._chart.clear()

    def __call__(self, x):
        return self.chart(x)[len(x)][0][self.S]

    def p_next(self, prefix):
        return self.next_token_weights(self.chart(prefix), prefix)

    def chart(self, prefix):
        c = self._chart.get(prefix)
        if c is None:
            c = self._compute_chart(prefix)
            self._chart[prefix] = c
        return c

    def _compute_chart(self, prefix):
        if len(prefix) == 0:
            tmp = [defaultdict(self.cfg.R.chart)]
            tmp[0][0][self.cfg.S] = self.nullary
            return tmp
        else:
            chart = self.chart(prefix[:-1])
            last_chart = self.extend_chart(chart, prefix)
            return chart + [
                last_chart
            ]  # TODO: avoid list addition here as it is not constant time!

    def next_token_weights(self, chart, prefix):
        """
        An O(N²) time algorithm to the total weight of a each next-token
        extension of `prefix`.
        """
        k = len(prefix) + 1

        cfg = self.cfg
        terminal = self.terminal
        r_y_xz = self.r_y_xz

        # the code below is just backprop / outside algorithm
        α = defaultdict(cfg.R.chart)
        α[0][cfg.S] += cfg.R.one

        # Binary rules
        for span in reversed(range(2, k + 1)):
            i = k - span
            α_i = α[i]
            for j in range(i + 1, k):
                chart_ij = chart[j][i]

                α_j = α[j]
                for Y, y in chart_ij.items():
                    for r in r_y_xz[Y]:
                        X = r.head
                        Z = r.body[1]
                        α_j[Z] += r.w * y * α_i[X]

        # Preterminal
        q = cfg.R.chart()
        tmp = α[k - 1]
        for w in cfg.V:
            for r in terminal[w]:
                q[w] += r.w * tmp[r.head]

        return q

    def extend_chart(self, chart, prefix):
        """
        An O(N²) time algorithm to extend to the `chart` with the last token
        appearing at the end of `prefix`; returns a new chart column.
        """
        k = len(prefix)

        cfg = self.cfg
        r_y_xz = self.r_y_xz

        new = defaultdict(cfg.R.chart)

        # Nullary
        new[k][cfg.S] += self.nullary

        # Preterminal
        tmp = new[k - 1]
        for r in self.terminal[prefix[k - 1]]:
            tmp[r.head] += r.w

        # Binary rules
        for span in range(2, k + 1):
            i = k - span
            new_i = new[i]
            for j in range(i + 1, k):
                chart_ij = chart[j][i]
                new_j = new[j]
                for Y, y in chart_ij.items():
                    for r in r_y_xz[Y]:
                        X = r.head
                        Z = r.body[1]
                        z = new_j[Z]
                        x = r.w * y * z
                        new_i[X] += x

        return new
