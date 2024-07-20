"""
Experimental method for sampling a tokenization that is consistent with an observed string.
"""

import numpy as np
import pylab as pl

from arsenal import colors
from arsenal.maths import sample_dict, logsumexp, compare
from collections import defaultdict

from genparse import Float


log_zero = float('-inf')


class PromptParticle:
    def __init__(self, lm, prompt, draw=sample_dict):
        context = ()
        k = 0

        V = sorted(lm.V)  # sorting for random seed consistency

        logP = 0.0
        logQ = 0.0
        logW = 0.0

        while True:
            logp = lm.logp_next(context)

            logq = defaultdict(lambda: log_zero)

            tmp = log_zero
            for y in V:
                if y == lm.eos:
                    continue
                if (
                    prompt[k:].startswith(y)
                    or len(prompt[k:]) < len(y)
                    and y.startswith(prompt[k:])
                ):
                    logq[y] = logp[y]
                    tmp = logsumexp([tmp, logp[y]])

            logπ = defaultdict(lambda: log_zero)
            for y in sorted(logq):
                logπ[y] = logq[y] - tmp

            y = draw.log_sample(logπ)

            logP += logp[y]
            logQ += logπ[y]
            logW += tmp

            context = context + (y,)

            k += len(y)

            if len(prompt[k:]) == 0:
                break

        self.context = context
        self.logP = logP
        self.logQ = logQ
        self.logW = logW

    def __repr__(self):
        return (
            f'{self.logW}:\t'
            + colors.light.cyan % '['
            + (colors.light.cyan % '|').join(
                # [colors.bg.magenta, '%s'][i % 2] % repr(y)[1:-1] for i, y in enumerate(self.context)
                repr(y)[1:-1]
                for i, y in enumerate(self.context)
            )
            + colors.light.cyan % ']'
        )


def p_next_healing(self, context):
    # token healing will take all but the last token and then resample the last one
    # since it might be a partial token.
    assert isinstance(context, str)
    token_ids = self.tokenizer.encode(context)
    tokens = tuple(self._decode[t] for t in token_ids)

    token_prefix = tokens[-1]
    tokens = tokens[:-1]

    _p = self.p_next(tokens)

    pp = Float.chart()
    for x in _p.keys():
        if x.startswith(token_prefix):
            pp[x] = _p[x]
    return pp.normalize()


def logprefix(self, context):
    return sum(self.logp_next(context[:i])[y] for i, y in enumerate(context))


import pandas as pd


def test_healing():
    from genparse.trace1 import TraceSWOR
    from genparse.util import set_seed, load_model_by_name
    from arsenal import iterview

    lm = load_model_by_name('gpt2')

    prompt = ' Sequential Monte Carlo is g'

    tracer = TraceSWOR()

    data = []

    for T in iterview(range(20)):
        with tracer:
            particle = PromptParticle(lm, prompt, draw=tracer)
            log_mass = tracer.cur.log_mass  # must do in the with-statement!

        print(f'{tracer.root.log_mass}\t', particle)

        data.append(
            {
                'context': particle.context,
                'logP': particle.logP,
                'logW': particle.logW,
                'logQ': particle.logQ,
                'logprefix': logprefix(lm, particle.context),
                'log_mass': log_mass,
            }
        )

    print(f'truncated with {tracer.root.log_mass:g} log mass left')
    df = pd.DataFrame(data)

    # identities about various quantities associated with the particle
    assert np.allclose(df.logP - df.logQ, df.logW)
    assert np.allclose(df.log_mass, df.logQ)

    # check that the target is the same as the logprefix weight
    assert np.allclose(df.logprefix, df.logP)

    # The proposal distribution over prompts is not well-correlated with the
    # prefix probability of the token string.
    assert not np.allclose(df.logprefix, df.logQ)

    # However, the importance weight distrbution is (we are using $p(x) =
    # \frac{p(x)}{q(x)} q(x) = w(x) q(x)$ here since we are enumerating $x$
    # rather than sampling it.)
    assert np.allclose(df.logprefix, df.logQ + df.logW)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
