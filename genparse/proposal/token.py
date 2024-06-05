from arsenal.maths import sample_dict
from arsenal import colors, timers
from arsenal.datastructures.pdict import pdict
from arsenal.iterextras import take

from genparse import Float
from genparse.proposal.trie import TokenCharacterTrie


# TODO: It's tempting to require proposal distributions to implement the `LM`
# interface, but it might be difficult to correctly implement `__call__` and
# `p_next` as proposal distributions may only be distributions over sample paths
# rather than character strings.  That appears to be a significant difference.

class TokenProposal(TokenCharacterTrie):
    """Proposal distribution that combines an `llm` and `guide`.  Let Y be the set
    of tokens and ∑ the set of characters.  We assume that llm is a distrbution
    over Y* and guide is a distribution over ∑*.

    We sample the next token y ∈ Y given ys ∈ Y* according the following
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

    def _p_next(self, context, K=None):
        with self.timer['llm']:
            p_llm = self.llm.p_next(self._prompt + context)

        with self.timer['cfg+trie']:
            return Float.chart(take(K, self.traverse_trie(context, p_llm))).normalize()

    def _update_internal(self):
        # overrides base method.  Takes max rather than sum of internal nodes
        jump = self.jump; mass = self.mass
        for node in self.ordering:
            m = 0
            for child in jump[node]:
                m = max(m, mass[child])
            mass[node] = m

    def traverse_trie(self, context, p_llm):
        """
        This method will lazily enumerate the nodes in the intersection of `p_llm` and
        and the `guide` for the given context.

        Here intersection means

          guide.p(token | context) * llm.p(token | context) for tokens ∈ llm.V

        """

        # update the trie with the llm's distribution of next token `p_llm`.
        self._update_trie(p_llm)

        agenda = pdict()
        P = Float.chart()

        # initial conditions
        (token, node) = ('', self.root)
        agenda[token, node] = 0
        P[node] = 1

        while agenda:

            (token, node) = agenda.pop()

            # Efficiently compute guide.p(x | context + token) for x ∈ guide.V.
            # These are individal characters that are aligned with the trie.
            p = self.guide.p_next(context + token)

            children_node = self.children[node]
            for x in children_node:

                if x is None:

                    #print(f'>>> {P[node] * self.mass[children_node[None]]:.20f} {token!r}')

                    yield (token, P[node] * self.mass[children_node[None]])
                    continue

                y = children_node[x]

                P[y] = P_y = P[node] * p[x]

                if P_y > 0:
                    agenda[token + x, y] = -P_y * self.mass[y]

    def sample(
        self,
        prompt = '',
        draw = sample_dict,
        chunked = False,
        max_tokens = float('inf'),
        verbosity = False,
        K = None,
    ):
        self._prompt = prompt
        context = ''
        chunks = []
        P = 1
        t = 0
        while True:
            t += 1
            if t <= max_tokens:
                p = self._p_next(context, K=K).normalize()
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
