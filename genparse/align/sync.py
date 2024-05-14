from arsenal.maths import sample_dict
from genparse import ERROR, SPACER, EOS
from genparse.util import normalize


# TODO: Replace this method with the local product of experts where one of the
# experts is a CharAlignedCFGLM.
def string_char_sync(lm1, char_lm2, draw=sample_dict, verbose=False, maxlen=100):
    context = ()
    prob = 1
    for _ in range(maxlen):
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


# XXX: This method has nothing to do with character-level models - it is just
# the local product of experts sampling method!
def char_char_sync(char_lm1, char_lm2, draw=sample_dict, maxlen=100):
    context = ()
    prob = 1
    for _ in range(maxlen):
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
