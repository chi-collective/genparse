from arsenal.maths import sample_dict


# TODO: Make the token-id sequences available as well as the character
# sequences.  Using the character sequences is useful the CFGLM caching, so we
# should not dispense with it!
class CharAlignedCFGLM:
    """
    This class implements a simple strategy for "aligning" a character-level
    CFG language model to a vocabulary of character chunks, such as those
    used in the common byte-pair encoding (BPE) schemes of large language models.
    """

    def __init__(self, lm, words, eos):

        # TODO: Correctly handle the possibility that the word model and lm may
        # have different EOS symbols. To accomodate this, we just we need
        # something that converts them that they don't have to be equal strings.
        # This will just amount to some special cases.
        assert eos in words

        self.lm = lm
        self.words = words
        self.eos = eos
        self._end = object()
        self.trie = self.make_trie(words)

#        self.c = {}
#        self._counts('', self.trie)

#    def _counts(self, prefix, node):
#        "Compute the number of continuations under `node`"
#        total = sum(
#            1 if x == self._end else self._counts(prefix + x, node[x])
#            for x in node
#        )
#        self.c[prefix] = total
#        return total

    def make_trie(self, words):
        root = dict()
        for word in words:
            curr = root
            for letter in word:
                curr = curr.setdefault(letter, {})
            curr[self._end] = self._end
        return root

    def p_next(self, context):
        t = len(context)
        return self.lm.cfg.R.chart(
            # strip the common length-t prefix
            (k[t:], v) for k,v in self.traverse_trie(context, self.trie, 1)
        ).normalize()

    def traverse_trie(self, context, node, P):
        p = self.lm.p_next(context)
        for x in node:
            if x == self._end:
                yield (context, P)
#                yield (context, self.lm.pfg(context))
                continue
            P_x = P * p[x]
            if P_x == 0: continue
            yield from self.traverse_trie(context + x, node[x], P_x)

    # TODO: test equivalence of `traverse_trie` and `traverse_naive`.
    def traverse_naive(self, context):
        for x in self.words:
            p_x = self.lm.pfg(context + x)  # prefix weight of context + x
            if p_x == 0: continue
            yield (context + x, p_x)

    # TODO: this should fall out of the base LM class
    def sample(self, draw=sample_dict, prob=False, chunked=False, verbose=False):
        context = ''
        chunks = []
        P = 1
        while True:
            if verbose: print(repr(context))
            p = self.p_next(context).normalize()
            y = draw(p)
            P *= p[y]
            if y == self.eos: break
            chunks.append(y)
            context += y
            # TODO: this is an ugly hack the arises from sloppy handling of EOS.
            # To handle this cleanly we just need to align the EOS in the LM and
            # the EOS in words.
            if context.endswith('</s>'): break
        if verbose: print(repr(context))
        value = context
        if chunked: value = tuple(chunks)
        if prob: value = (value, P)
        return value
