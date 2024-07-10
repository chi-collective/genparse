from arsenal import colors

from arsenal.datastructures import LocatorMaxHeap
from arsenal.iterextras import take
from arsenal.maths import sample_dict

from genparse.proposal.trie import TokenCharacterTrie
from genparse.semiring import Float


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

    where φ: Y* → ∑* maps token strings to character strings.

    """

    def __init__(self, *, llm, guide, K=None):
        self.llm = llm
        self.guide = guide
        self._prompt = None
        self._p_guide = None
        self.K = K

        # Filter LLM tokens that are illegal under the cfg
        words = {word for word in llm.V if set(word) <= self.guide.V or word == llm.eos}

        super().__init__(words, old_eos=llm.eos, new_eos=guide.eos)

    def _p_next(self, context, K=None, execute_model_req=None, **kwargs):  # pylint: disable=unused-argument
        p_llm = self.llm.p_next(
            self._prompt + context, execute_model_req=execute_model_req
        )
        return Float.chart(take(K, self.traverse_trie(context, p_llm))).normalize()

    async def sample_next_token(
        self,
        prompt,
        context,
        draw=sample_dict,
        p_llm=None,
        **kwargs,
    ):  # pylint: disable=unused-argument
        r"""
        Proposes a token and incremental weight update.

        The following procedure, justified using RAVI, gives the way we sample a token and compute the incremental SMC weight update.

        1. Sample a subset S of size K (+ 1) of the token vocabulary by
            a. enumerating the top K tokens
            b. if there are remaining tokens with non-zero probability, sampling a wilcard token from the remainder of the vocabulary proportional to p_llm(x)
                * this step ensures absoluate continuity
        2. Compute *unnormalized target* p(x) of each x \in S according to p_llm(x)p_cfg(x).
        3. Compute (local) weight w(x) of each token as p(x)/Pr(x \in S) where Pr(x \in S) is the *inclusion probability*.
            * Pr(x \in S) = 1 if x in top K
            * Pr(x \in S) \propto p_llm(x) for the wilcard token, if applicable
        4. Renormalize the weights of the tokens in S and sample one of them.
        5. Set the incremental SMC weight update w'(x) = \sum_{x \in S} w(x)

        Args:
            prompt : The LLM prompt.
            context : The previous generated tokens.
            p_llm: Provide the model with pre-computed p_llm. Since for VLLM, p_llm is computed
                for all particles altogether. We directly pass the corresponding p_llm to
                the proposal of each particle.
        Returns:
            token : Proposed LLM token.
            weight_update : Incremental SMC weight update.

        """
        proposal_p = 1

        if p_llm is None:
            p_llm = await self.llm.p_next_async(prompt + context)

        # enumerate top K tokens
        Ws = Float.chart(take(self.K, self.traverse_trie(context, p_llm)))

        # Was the distrbution truncated?  If so, use a wildcard sample to debias it.
        if self.K is not None and len(Ws) == self.K:
            # compute distribution over wildcard tokens
            P_wc = Float.chart({x: p for x, p in p_llm.items() if x not in Ws and p > 0})

            # if P_wc is non-empty, sample a wildcard token to ensure absolute continuity
            if P_wc:
                P_wc = P_wc.normalize()
                wildcard = draw(P_wc)
                proposal_p *= P_wc[wildcard]

                # compute the wildcard's weight
                p_cfg_wc = self.guide.p_next_seq(''.join(context), wildcard)

                Ws[wildcard] = (
                    p_llm[self.old_eos if wildcard == self.new_eos else wildcard]
                    * p_cfg_wc
                    / P_wc[wildcard]
                )

        Ws = Ws.trim()

        if Ws:
            # sample token from weights and compute update
            Ws_norm = Ws.normalize()
            token = draw(Ws_norm)
            proposal_p *= Ws_norm[token]
            weight_update = Ws.sum()
        else:
            # if there are no possible next tokens, kill the particle
            token = '💀'
            weight_update = 0

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
        assert isinstance(context, tuple), context
        assert set(context) <= self.llm.V, f'OOV detected {set(context) - self.llm.V}'

        # update the trie with the llm's distribution of next token `p_llm`.
        self.mass[:] = 0  # reset the mass
        self._update_leaves(p_llm)

        h = self.mass.copy()

        # Update internal nodes of our A* heuristic
        jump = self.jump
        for node in self.ordering:
            m = 0
            for child in jump[node]:
                m = max(m, h[child])
            h[node] = m

        agenda = LocatorMaxHeap()

        P = Float.chart()

        # initial conditions
        (token, node) = ('', self.root)
        agenda[token, node, False] = 1
        P[node] = 1

        self._p_guide = {}
        children = self.children

        curr_priority = 1
        prev_best = 1
        while agenda:
            (token, node, done), score = agenda.popitem()

            assert score <= curr_priority
            curr_priority = score

            # terminal state
            if done:
                value = P[node] * h[node]
                assert prev_best >= value
                prev_best = value
                self._p_guide[token] = P[node]
                yield (token, value)
                continue

            # Efficiently compute guide.p(x | context + token) for x ∈ guide.V.
            # These are individual characters that are aligned with the trie.
            p = self.guide.p_next(''.join(context) + token)

            for x, y in children[node].items():
                if x is None:
                    P_y = P[node]
                    P[y] = P_y
                    agenda[token, y, True] = P_y * h[y]

                else:
                    P_y = P[node] * p[x]
                    if P_y == 0:
                        continue
                    P[y] = P_y
                    agenda[token + x, y, False] = P_y * h[y]

    def sample(
        self,
        prompt=(),
        draw=sample_dict,
        chunked=False,
        max_tokens=float('inf'),
        verbosity=False,
        K=None,
    ):
        self._prompt = prompt
        context = ()
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
            context = context + (y,)
            if verbosity > 0:
                print(colors.cyan % y, end=colors.magenta % '|')
        value = context
        if chunked:
            value = tuple(chunks)
        if verbosity > 0:
            print()
        return (value, P)
