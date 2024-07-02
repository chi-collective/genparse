from arsenal import colors, timers
from arsenal.maths import sample_dict

from genparse.proposal.trie_numba import TokenCharacterTrie
from genparse.semiring import Float


class CharacterProposal(TokenCharacterTrie):
    """Proposal distribution that combines an `llm` (token-based LM) and `guide`
    (character-based LM).

    The way that samples are generated is that we
    (1) materialize the next-token distribution `llm.p_next(context)`
    (2) convert it into a character-level trie augmented with an end-of-token marker.
    (3) sample a path in the trie (starting at its root) which takes the local
        product of the trie distribution and the guide, excluding the
        end-of-token.
    (4) given the path, we then sample an end-of-token anywhere along the path.

    The reason why we like this proposal distribution is its efficiency: in
    practice, `p_llm` is one big batched evaluation, that is given by a blackbox
    model, and `p_guide` is a character-level LM.  Although, any given call to
    p_guide is fast, calling it for every token is very slow - even with GPU
    parallelism.  This proposal distrbution avoid making a huge number of calls
    to p_guide (as in `CharAlignedCFGLM`) by *sampling* paths in the
    character-trie rather than *enumerating* them.

    We could probably improve this generative procees by collapsing the
    post-path sampling of exits, but it would probably require the cost that we
    are trying to avoid!  (That is probably deeply connected with
    `CharAlignedCFGLM`, but we haven't worked out the precise connection.)

    """

    # pylint: disable=redefined-slots-in-subclass
    __slots__ = TokenCharacterTrie.__slots__ + (
        'llm',
        'guide',
        'timer',
    )

    def __init__(self, *, llm, guide):
        self.llm = llm
        self.guide = guide
        self.timer = timers()

        # Filter LLM tokens that are illegal under the cfg
        words = {word for word in llm.V if set(word) <= self.guide.V or word == llm.eos}

        super().__init__(words, encode=llm._encode, old_eos=llm.eos, new_eos=guide.eos)

    def sample(
        self, prompt, max_tokens=float('inf'), verbosity=0, draw=sample_dict, **kwargs
    ):
        context = ''
        W = 1
        P = 1
        t = 0
        while True:
            t += 1
            if t <= max_tokens:
                with self.timer['llm'](t=len(context)):
                    p_llm = self.llm.p_next(prompt + context)
                with self.timer['cfg+trie'](t=len(context)):
                    self._update_trie(p_llm)
                    token, proposal_p, weight_update = self._guided_sample_trie(
                        context, verbosity=verbosity, draw=draw, **kwargs
                    )
            else:
                token = self.guide.eos
                weight_update = 1
                proposal_p = 1
            W *= weight_update
            P *= proposal_p
            if self.guide.eos == token:
                break
            if verbosity > 0:
                print(colors.cyan % token, end=colors.magenta % '|')
            context += token
        if verbosity > 0:
            print()
        return (context, P, W)

    async def sample_next_token(
        self,
        prompt,
        context,
        verbosity=0,
        correct_weights=True,
        draw=sample_dict,
        p_llm=None,
        **kwargs,
    ):
        """
        Proposes a token and incremental weight update.

        Args:
            prompt : The LLM prompt.
            context : The previous generated tokens.
            verbosity : > 1 prints sampling process.
            correct_weights : Whether to correct the importance weights with RAVI.
                false leads to probabilistically incorrect inference.
            p_llm: Provide the model with pre-computed p_llm. Since for VLLM, p_llm is computed
                for all particles altogether. We directly pass the corresponding p_llm to
                the proposal of each particle.
        Returns:
            token : Proposed LLM token.
            weight_update : Incremental SMC weight update.
        """
        if p_llm is None:
            with self.timer['llm'](t=len(context)):
                p_llm = await self.llm.p_next_async(prompt + context)

        self._update_trie(p_llm)

        with self.timer['cfg+trie'](t=len(context)):
            if correct_weights:
                (token, proposal_p, weight_update) = self._guided_sample_trie(
                    context, draw=draw, verbosity=verbosity, **kwargs
                )
            else:
                (token, proposal_p, weight_update) = self._guided_sample_trie_uncorrected(
                    context, draw=draw, verbosity=verbosity, **kwargs
                )

        return (token, proposal_p, weight_update)

    def __deepcopy__(self, memo):
        cpy = type(self).__new__(type(self))

        # the only thing that needs a real copy is the mass array
        cpy.mass = self.mass.copy()

        # pass the other member variables thru
        cpy.root = self.root
        cpy.children = self.children
        cpy.word2leaf = self.word2leaf
        cpy.jump = self.jump
        cpy.ordering = self.ordering
        cpy.llm = self.llm
        cpy.guide = self.guide
        cpy.timer = self.timer
        cpy.old_eos = self.old_eos
        cpy.new_eos = self.new_eos
        cpy.token_id_to_leaf = self.token_id_to_leaf

        return cpy

    def _guided_sample_trie(self, context, draw, verbosity=0):
        """
        This function samples a token from the trie and computes the incremental weight update.

        The following procedure, justified using RAVI, gives the way we sample a token and compute the incremental SMC weight update.

            1. Sample a subset $S$ of the token vocabulary by sampling a path through the trie.
            2. Compute *unnormalized target* $p(x)$ of each $x \in S$ according to $p_\text{LLM}(x)p_\text{CFG}(x)$.
                * $p_\text{LLM}(x)$ is given from the mass at the leaf of the trie;
                * $p_\text{CFG}(x)$ is given as the product of the next character distributions up to that point in the path
            3. Compute (local) weight $w(x)$ of each token as $\frac{p(x)}{\Pr(x \in S)}$ where $\Pr(x \in S)$ is the *inclusion probability*.
                * $\Pr(x \in S)$ in the character proposal is given as the probability of the path prefix up to $x$.
            4. Renormalize the weights of the tokens in $S$ and sample one of them.
            5. Set the incremental SMC weight update $w^\prime(x) = \sum_{x \in S} w(x)$

        """
        curr = self.root
        path = []

        inclusion_prob = 1  # path prefix probability
        cfg_prob = 1
        proposal_p = 1  # probability of trace

        weights = Float.chart()

        children = self.children
        mass = self.mass

        if verbosity > 1:
            print(colors.line(80))
        while True:
            children_curr = children[curr]
            mass_curr = mass[curr]

            p1 = Float.chart((a, mass[c] / mass_curr) for a, c in children_curr.items())

            p2 = self.guide.p_next(context + ''.join(path)).trim()

            if None in p1:
                token = ''.join(path)

                weights[token] = (mass[children_curr[None]] * cfg_prob) / inclusion_prob

                if verbosity > 1:
                    print(
                        colors.blue % 'ADDED TOKEN TO S',
                        repr(token),
                        'weight=',
                        weights[token],
                        'token prob=',
                        mass[children_curr[None]] * cfg_prob,
                        'inclusion prob=',
                        inclusion_prob,
                    )

            _q = (p1 * p2).trim()

            if verbosity > 1:
                print(colors.yellow % 'calling context=', repr(''.join(context)))
                print(colors.yellow % 'partial token=', repr(''.join(path)))
                if not _q:
                    print('llm (top 10) =', p1.top(10))
                    print('guide (top 10) =', p2.top(10))
                print('_q (top 10) =', _q.top(10))

            if not _q:
                break

            q = _q.normalize()

            a = draw(q)
            inclusion_prob *= q[a]
            cfg_prob *= p2[a]
            proposal_p *= q[a]

            curr = children_curr[a]

            if verbosity > 1:
                print(colors.orange % 'action', repr(a), 'context', repr(''.join(path)))

            path.append(a)

        normalized_weights = weights.normalize()

        if verbosity > 1:
            print(colors.light.green % 'token weights:', weights)
            print(colors.light.green % 'token probs:', normalized_weights)

        token = draw(normalized_weights)
        proposal_p *= normalized_weights[token]
        weight_update = weights.sum()

        if verbosity > 1:
            print(colors.orange % 'sampled token=', repr(token))
            print(colors.orange % 'weight update=', weight_update)

        return (token, proposal_p, weight_update)

    def _guided_sample_trie_uncorrected(self, context, draw, verbosity=0):
        """
        This function samples a token from the trie and computes the incremental weight update.

        WARNING: This function is probabilistically incorrect; it produces biased estimates.
        The returned weight update is given as p_llm(x) * p_cfg(x) / q(x,S) where x is the proposed token
        and S is the path through the trie from which x is sampled.

        """
        curr = self.root
        path = []
        guide_prob = 1
        proposal_prob = 1
        exits = Float.chart()

        children = self.children
        mass = self.mass

        if verbosity > 1:
            print(colors.line(80))
        while True:
            children_curr = children[curr]
            mass_curr = mass[curr]

            p1 = Float.chart((a, mass[c] / mass_curr) for a, c in children_curr.items())

            p2 = self.guide.p_next(context + ''.join(path)).trim()

            if None in p1:
                exits[''.join(path)] = mass[children_curr[None]]
                if verbosity > 1:
                    print(
                        colors.blue % 'ADDED EXIT',
                        repr(''.join(path)),
                        'prob=',
                        proposal_prob,
                    )

            _q = (p1 * p2).trim()

            if verbosity > 1:
                print(colors.yellow % 'calling context=', repr(''.join(context)))
                print(colors.yellow % 'partial token=', repr(''.join(path)))
                if not _q:
                    print('llm (top 10) =', p1.top(10))
                    print('guide (top 10) =', p2.top(10))
                print('_q (top 10) =', _q.top(10))

            if not _q:
                break

            q = _q.normalize()

            a = draw(q)
            guide_prob *= p2[a]
            proposal_prob *= q[a]
            curr = children_curr[a]

            if verbosity > 1:
                print(colors.orange % 'action', repr(a), 'context', repr(''.join(path)))

            path.append(a)

        # Sample the end-of-token marker in hindsight
        exits = exits.normalize()

        if verbosity > 1:
            print(colors.light.green % 'p exits:', exits)

        token = draw(exits)

        if verbosity > 1:
            print(colors.orange % 'picked exit', repr(path))

        proposal_prob *= exits[token]

        llm_prob = mass[self.word2leaf[token]]

        weight_update = (llm_prob * guide_prob) / proposal_prob

        return (token, proposal_prob, weight_update)
