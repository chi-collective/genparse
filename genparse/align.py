from arsenal import colors
from arsenal.maths import sample_dict
from collections import Counter, defaultdict, deque

from genparse import Float, Chart, EOS
from genparse.cfglm import SPACER
from genparse.steer import normalize

ERROR = '💥'

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


def string_char_sync(lm1, char_lm2, draw=sample_dict, verbose=False):
    context = ()
    prob = 1
    for _ in range(100):
        p1 = lm1.p_next(context)
        if verbose: print('predict', context, '::')
        p12 = {}
        for x in p1:
            if x == EOS:
                char_context = SPACER.join(context) + SPACER
                p_char = char_lm2.p_next(char_context)[EOS]
                p12[x] = p1[x] * p_char
                if verbose: print(' ', x, p1[x], char_context, '::', EOS, p_char)
                continue
            char_context = SPACER.join(context + (x,))
            char = SPACER
            p_char = char_lm2.p_next(char_context)[SPACER]
            if verbose: print(' ', x, p1[x], char_context, '::', char, p_char)
            p12[x] = p1[x] * p_char
        if sum(p12.values()) == 0:
            context = context + (ERROR,)
            break
        p12 = normalize({a: p for a,p in p12.items() if p > 0})
        x = draw(p12)
        prob *= p12[x]
        context = context + (x,)
        if x == EOS: break
    return SPACER.join(context), prob


def char_char_sync(char_lm1, char_lm2, draw=sample_dict):
    context = ()
    prob = 1
    for _ in range(100):
        p1 = char_lm1.p_next(context)
        p2 = char_lm2.p_next(context)
        p12 = {x: p1[x] * p2[x] for x in p1}
        if sum(p12.values()) == 0:
            context = context + (ERROR,)
            break
        p12 = normalize({a: p for a,p in p12.items() if p > 0})
        x = draw(p12)
        prob *= p12[x]
        context = context + (x,)
        if x == EOS: break
    return ''.join(context), prob
