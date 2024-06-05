from arsenal.maths import sample_dict
from arsenal import colors, timers
from genparse import Float
from genparse.proposal.trie import TokenCharacterTrie


# TODO: It's tempting to require proposal distributions to implement the `LM`
# interface, but it might be difficult to correctly implement `__call__` and
# `p_next` as proposal distributions may only be distributions over sample paths
# rather than character strings.  That appears to be a significant difference.

# TODO: Make the token-id sequences available as well as the character
# sequences.  Using the character sequences is useful the CFGLM caching, so we
# should not dispense with it!


class TokenProposal(TokenCharacterTrie):
    """Proposal distribution that combines an `llm` and `guide`.  Let Y be the set
    of tokens and ∑ the set of characters.  We assume that llm is a distrbution
    over Y* and guide is a distribution over ∑*.

    We sample the next token y ∈ Y given ys ∈Y* according the following
    distrbution:

      q(y | ys) ∝ p_llm(y | ys) * p_guide(φ(y) | φ(ys))

    where φ: Y* → ∑* maps token strings to characters.

    """

    def __init__(self, *, llm, guide):
        self.llm = llm
        self.guide = guide
        self.V = llm.V
        self._prompt = None
        self.timer = timers()

        # Filter LLM tokens that are illegal under the cfg
        words = {
            word
            for word in llm.V
            if set(word) <= self.guide.V or word == llm.eos
        }

        super().__init__(words, old_eos = llm.eos, new_eos = guide.eos)

    def _p_next(self, context):
        with self.timer['llm']:
            p_llm = self.llm.p_next(self._prompt + context)

        with self.timer['cfg+trie']:
            self._update_trie(p_llm)
            return Float.chart(self.traverse_trie(context, '', self.root, 1)).normalize()

#    def traverse_trie(self, context, token, node, P):
#        p = self.guide.p_next(context)
#        children_node = self.children[node]
#        for x in children_node:
#            if x is None:
#                yield (token, P * children_node[None])
#                continue
#            P_x = P * p[x]
#            if P_x == 0: continue
#            yield from self.traverse_trie(context + x, token + x, children_node[x], P_x)

    def traverse_trie(self, context, token, node, P):

        agenda = [(context, token, node, P)]

        while agenda:

            #item = max(agenda, key = lambda x: x[3] * self.mass[x[2]])
            #agenda.remove(item)
            item = agenda.pop()

            (context, token, node, P) = item

            p = self.guide.p_next(context)

            children_node = self.children[node]
            for x in children_node:

                if x is None:
                    yield (token, P * self.mass[self.children[node][None]])
                    continue

                P_x = P * p[x]

                if P_x > 0:
                    agenda.append((context + x, token + x, children_node[x], P_x))

    def sample(self, prompt='', draw=sample_dict, chunked=False, max_tokens=float('inf'), verbosity=False):
        self._prompt = prompt
        context = ''
        chunks = []
        P = 1
        t = 0
        while True:
            t += 1
            if t <= max_tokens:
                p = self._p_next(context).normalize()
                y = draw(p)
                P *= p[y]
            else:
                y = self.guide.eos
                P *= 1   # deterministic
            if y == self.guide.eos: break
            chunks.append(y)
            context += y
            if verbosity > 0: print(colors.cyan % y, end=colors.magenta % '|')
        value = context
        if chunked: value = tuple(chunks)
        if verbosity > 0: print()
        #self.timer.compare()
        return (value, P)
