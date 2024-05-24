from arsenal.maths import sample_dict
from genparse import Float


# TODO: It's tempting to require proposal distributions to implement the `LM`
# interface, but it might be different to correctly implement `__call__` and
# `p_next` as proposal distributions may only be distributions over sample paths
# rather than character strings.  That appears to be a significant difference.

# TODO: Make the token-id sequences available as well as the character
# sequences.  Using the character sequences is useful the CFGLM caching, so we
# should not dispense with it!
class TokenProposal:
    """
    This class implements a simple strategy for "aligning" a character-level
    CFG language model to a vocabulary of character chunks, such as those
    used in the common byte-pair encoding (BPE) schemes of large language models.
    """

    def __init__(self, *, llm, guide):

        self.llm = llm
        self.guide = guide

        self.V = llm.V

        self._end = object()
        self.trie = self.make_trie(self.V)
        self._p_llm = None

    def make_trie(self, words):
        root = dict()
        for word in words:
            if word == self.llm.eos: word = self.guide.eos
            curr = root
            for letter in word:
                curr = curr.setdefault(letter, {})
            curr[self._end] = self._end
        return root

    def p_next(self, context):
        t = len(context)
        self._p_llm = self.llm.p_next(context)
        self._p_llm[self.guide.eos] = self._p_llm[self.llm.eos]
        return Float.chart(
            # strip the common length-t prefix
            (k[t:], v) for k,v in self.traverse_trie(context, '', self.trie, 1)
        ).normalize()

    def traverse_trie(self, context, token, node, P):
        p = self.guide.p_next(context)
        for x in node:
            if x == self._end:
                yield (context, P * self._p_llm[token])
                continue
            P_x = P * p[x]
            if P_x == 0: continue
            yield from self.traverse_trie(context + x, token + x, node[x], P_x)

    def sample(self, draw=sample_dict, chunked=False, verbose=False):
        context = ''
        chunks = []
        P = 1
        while True:
            if verbose: print(repr(context))
            p = self.p_next(context).normalize()
            y = draw(p)
            P *= p[y]
            if y == self.guide.eos: break
            chunks.append(y)
            context += y
        if verbose: print(repr(context))
        value = context
        if chunked: value = tuple(chunks)
        return (value, P)
