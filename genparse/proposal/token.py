from arsenal import colors, timers
from arsenal.datastructures.pdict import pdict
from arsenal.iterextras import take
from arsenal.maths import sample_dict

from genparse.proposal.trie import TokenCharacterTrie
from genparse.semiring import Float

from inspect import iscoroutinefunction

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

    def __init__(self, *, llm, guide, K=None):
        self.llm = llm
        self.guide = guide
        self._prompt = None
        self._p_guide = None
        self.timer = timers()
        self.K = K

        # Filter LLM tokens that are illegal under the cfg
        words = {word for word in llm.V if set(word) <= self.guide.V or word == llm.eos}

        super().__init__(words, old_eos=llm.eos, new_eos=guide.eos)

    def _p_next(self, context, K=None, execute_model_req=None, **kwargs):
        p_llm = self.llm.p_next(
            self._prompt + context, execute_model_req=execute_model_req
        )
        return Float.chart(take(K, self.traverse_trie(context, p_llm))).normalize()

    async def sample_next_token(
        self,
        prompt,
        context,
        verbosity=0,
        draw=sample_dict,
        p_llm=None,
        **kwargs,
    ):
        """
        Proposes a token and incremental weight update.

        The following procedure, justified using RAVI, gives the way we sample a token and compute the incremental SMC weight update.

        1. Sample a subset S of size K of the token vocabulary by
            a. enumerating the top K - 1 tokens
            b. sampling a wilcard token from the remainder of the vocabulary proportional to p_llm(x)
        2. Compute *unnormalized target* p(x) of each x \in S according to p_llm(x)p_cfg(x).
        3. Compute (local) weight w(x) of each token as p(x)/Pr(x \in S) where Pr(x \in S) is the *inclusion probability*.
            * Pr(x \in S) = 1 if x in top K - 1
            * Pr(x \in S) \propto p_llm(x) for the wilcard token
        4. Renormalize the weights of the tokens in S and sample one of them.
        5. Set the incremental SMC weight update w'(x) = \sum_{x \in S} w(x)

        Args:
            prompt : The LLM prompt.
            context : The previous generated tokens.
            verbosity : > 1 prints sampling process.
            p_llm: Provide the model with pre-computed p_llm. Since for VLLM, p_llm is computed
                for all particles altogether. We directly pass the corresponding p_llm to
                the proposal of each particle.
        Returns:
            token : Proposed LLM token.
            weight_update : Incremental SMC weight update.

        """
        if p_llm is None:
            with self.timer['llm'](t=len(context)):
                if iscoroutinefunction(self.llm.p_next):
                    p_llm = await self.llm.p_next(
                        prompt + context
                    )
                else:
                    p_llm = self.llm.p_next(
                        prompt + context
                    )

        # enumerate top K - 1 tokens
        Ws = Float.chart(take(self.K - 1, self.traverse_trie(context, p_llm)))

        # sample wildcard token from p_llm
        P_wc = Float.chart({x: p for x, p in p_llm.items() if x not in Ws}).normalize()
        wildcard = draw(P_wc)
        proposal_p = P_wc[wildcard]

        # compute wild card weight
        p_cfg_wc = 1
        with self.timer['cfg+trie'](t=len(context)):
            for i, c in enumerate(wildcard):
                p_cfg_wc *= self.guide.p_next(context + wildcard[:i])[c]
        Ws[wildcard] = (
            p_llm[self.old_eos if wildcard == self.new_eos else wildcard]
            * p_cfg_wc
            / P_wc[wildcard]
        )

        # sample token from weights and compute update
        Ws_norm = Ws.normalize()
        token = draw(Ws_norm)
        proposal_p *= Ws_norm[token]
        weight_update = Ws.sum()

        return (token, proposal_p, weight_update)

    def _update_internal(self):
        # overrides base method.  Takes max rather than sum of internal nodes
        jump = self.jump
        mass = self.mass
        for node in self.ordering:
            m = 0
            for child in jump[node]:
                m = max(m, mass[child])
            mass[node] = m

    def __deepcopy__(self, memo):
        cpy = type(self).__new__(type(self))

        # the only thing that needs a real copy is the mass array
        cpy.mass = self.mass.copy()
        cpy._p_guide = self._p_guide if self._p_guide is None else self._p_guide.copy()

        # pass the other member variables thru
        cpy.root = self.root
        cpy.children = self.children
        cpy.word2leaf = self.word2leaf
        cpy.jump = self.jump
        cpy.ordering = self.ordering
        # cpy.token_id_to_leaf = self.token_id_to_leaf    # TODO: when we switch to the numba version
        cpy.llm = self.llm
        cpy.guide = self.guide
        cpy.timer = self.timer
        cpy.old_eos = self.old_eos
        cpy.new_eos = self.new_eos
        cpy._prompt = self._prompt
        cpy.K = self.K

        return cpy

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

        self._p_guide = {}

        while agenda:
            (token, node) = agenda.pop()

            # Efficiently compute guide.p(x | context + token) for x ∈ guide.V.
            # These are individal characters that are aligned with the trie.
            p = self.guide.p_next(context + token)

            children_node = self.children[node]
            for x in children_node:
                if x is None:
                    # print(f'>>> {P[node] * self.mass[children_node[None]]:.20f} {token!r}')

                    self._p_guide[token] = P[node]

                    yield (token, P[node] * self.mass[children_node[None]])
                    continue

                y = children_node[x]

                P[y] = P_y = P[node] * p[x]

                if P_y > 0:
                    agenda[token + x, y] = -P_y * self.mass[y]

    def sample(
        self,
        prompt='',
        draw=sample_dict,
        chunked=False,
        max_tokens=float('inf'),
        verbosity=False,
        K=None,
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
                P *= 1  # deterministic
            if y == self.guide.eos:
                break
            chunks.append(y)
            context += y
            if verbosity > 0:
                print(colors.cyan % y, end=colors.magenta % '|')
        value = context
        if chunked:
            value = tuple(chunks)
        if verbosity > 0:
            print()
        # self.timer.compare()
        return (value, P)
