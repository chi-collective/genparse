from arsenal import colors
from arsenal.maths import sample_dict
from collections import Counter, deque

from genparse import Float, Chart, EOS, SPACER, ERROR
from genparse.util import normalize


# TODO: use A^* to break ties (ambiguity); order fringe by probability
def pullback(lm, qs, decode=''.join, init=(), verbose=False):
    """
    Brute-force tokenization alignment; returns posterior over next token that may follow `qs`
    under `lm`.  However, `lm` uses a different tokenization that we need to `decode` into the
    domain of `qs`.
    """

    #from genparse.wfsa import WFSA
    #m = WFSA()
    #m.add_I((), 1)

    total = Counter()
    k = len(qs)
    Q = deque([(1, init)])
    while Q:
        (p, ys) = Q.popleft()

        xs = decode(ys)

        info = (repr(qs), repr(xs), [decode([y]) for y in ys], ys, p)

        if len(xs) > k:

            if xs.startswith(qs):
                total[xs[k]] += p
                if verbose: print(colors.green % 'summing', info)
                continue

            else:
                continue

        if xs != qs[:len(xs)]:
            continue

        if verbose: print('expanding', info)

        #m.add_F(ys, p)
        assert xs == qs[:len(xs)]

        # `xs` is incomplete but agrees with `qs` so far, so we continue to expand it

        distribution = lm.p_next(ys)

        if hasattr(distribution, 'items'):
            distribution = distribution.items()
        else:
            #print('thing2')
            distribution = list(enumerate(distribution.numpy()))

        #print(distribution)

        for y, next_p in distribution:     # TODO: we can filter here instead of at pop time
            Q.append((next_p, ys + (y,)))

        #m.add_arc(ys, y, ys + (y,), next_p/p)

    #display(m)
    #display(m.min.threshold(1e-8))

    return Chart(Float, total)


