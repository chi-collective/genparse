from arsenal.maths import sample_dict
from arsenal import colors
from genparse import Float


# TODO: It's tempting to require proposal distributions to implement the `LM`
# interface, but it might be different to correctly implement `__call__` and
# `p_next` as proposal distributions may only be distributions over sample paths
# rather than character strings.  That appears to be a significant difference.

# TODO: Make the token-id sequences available as well as the character
# sequences.  Using the character sequences is useful the CFGLM caching, so we
# should not dispense with it!
class TokenProposal:
    """Proposal distribution that combines an `llm` and `guide`.  Let Y be the set
    of tokens and ∑ the set of characters.  We assume that llm is a distrbution
    over Y* and guide is a distribution over ∑*.

    We sample the next token y ∈ Y given ys ∈Y* according the following
    distrbution:

      q(y | ys) \propto p_llm(y | ys) * p_guide(φ(y) | φ(ys))

    where φ: Y* → ∑* maps token strings to characters.

    """

    def __init__(self, *, llm, guide):
        self.llm = llm
        self.guide = guide
        self.V = llm.V
        self._end = None
        self.trie = self.make_trie(self.V)
        self._p_llm = None
        self._prompt = None

    def make_trie(self, words):
        root = {}
        for word in words:
            if word == self.llm.eos: word = self.guide.eos
            curr = root
            for letter in word:
                curr = curr.setdefault(letter, {})
            curr[self._end] = self._end
        return root

    def _p_next(self, context):
        self._p_llm = self.llm.p_next(self._prompt + context)
        self._p_llm[self.guide.eos] = self._p_llm[self.llm.eos]
        return Float.chart(self.traverse_trie(context, '', self.trie, 1)).normalize()

    def traverse_trie(self, context, token, node, P):
        p = self.guide.p_next(context)
        for x in node:
            if x == self._end:
                yield (token, P * self._p_llm[token])
                continue
            P_x = P * p[x]
            if P_x == 0: continue
            yield from self.traverse_trie(context + x, token + x, node[x], P_x)

    def sample(self, prompt='', draw=sample_dict, chunked=False, max_tokens=float('inf'), verbosity=False):
        self._prompt = prompt
        context = ''
        chunks = []
        P = 1
        t = 0
        while True:
            t += 1
            p = self._p_next(context).normalize()
            y = draw(p) if t <= max_tokens else self.guide.eos
            P *= p[y]
            if y == self.guide.eos: break
            chunks.append(y)
            context += y
            if verbosity > 0: print(colors.cyan % y, end=colors.magenta % '|')
        value = context
        if chunked: value = tuple(chunks)
        if verbosity > 0: print()
        return (value, P)
