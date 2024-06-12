import numpy as np
from arsenal.maths import sample_dict
from arsenal import colors, timers

from genparse import Float
from genparse.proposal.trie import TokenCharacterTrie


class CharacterProposal(TokenCharacterTrie):
    """
    Proposal distribution that combines an `llm` (token-based LM) and `guide` (character-based LM).

    The way that samples are generated is that we
    (1) materialize the next-token distribution `llm.p_next(context)`
    (2) convert it into a character-level trie augmented with an end-of-token marker.
    (3) sample a path in the trie (starting at its root) which takes the local product of the
        trie distribution and the guide, excluding the end-of-token.
    (4) given the path, we then sample an end-of-token anywhere along the path.

    The reason why we like this proposal distribution is its efficiency: in practice, `p_llm` is one big
    batched evaluation, that is given by a blackbox model, and `p_guide` is a CFGLM.  Although, any given
    call to p_guide is fast, calling it for every token is very slow - even with GPU parallelism.  This
    proposal distrbution avoid making a huge number of calls to p_guide (as in `CharAlignedCFGLM`) by
    *sampling* paths in the character-trie rather than *enumerating* them.

    We could probably improve this generative procees by collapsing the post-path sampling of exits,
    but it would probably require the cost that we are trying to avoid!  (That is probably deeply connected
    with `CharAlignedCFGLM`, but we haven't worked out the precise connection.)

    """

    __slots__ = (
        'root',
        'children',
        'mass',
        'word2leaf',
        'jump',
        'ordering',
        'llm',
        'guide',
        'timer',
        'old_eos',
        'new_eos',
    )

    def __init__(self, *, llm, guide):
        self.llm = llm
        self.guide = guide
        self.timer = timers()

        # Filter LLM tokens that are illegal under the cfg
        words = {
            word
            for word in llm.V
            if set(word) <= self.guide.V or word == llm.eos
        }

        super().__init__(words, old_eos = llm.eos, new_eos = guide.eos)

    def sample(self, prompt, max_tokens=float('inf'), verbosity=0, **kwargs):
        context = ''
        P = 1
        t = 0
        while True:
            t += 1
            if t <= max_tokens:
                with self.timer['llm'](t=len(context)):
                    p_llm = self.llm.p_next(prompt + context)
                with self.timer['cfg+trie'](t=len(context)):
                    self._update_trie(p_llm)
                    token, p_token, _, _ = self._guided_sample_trie(
                        self.root, context, verbosity=verbosity, **kwargs
                    )
            else:
                token = self.guide.eos
                p_token = 1
            P *= p_token
            if self.guide.eos == token: break
            if verbosity > 0: print(colors.cyan % token, end=colors.magenta % '|')
            context += token
        if verbosity > 0: print()
        self.timer.compare()
        return (context, P)

    async def sample_next_token(self, prompt, context, verbosity=0, compare_time=False, **kwargs):
        with self.timer['llm'](t=len(context)):
            p_llm = await self.llm.p_next(prompt + context)
        with self.timer['cfg+trie'](t=len(context)):
            self._update_trie(p_llm)
            (path, llm_prob, guide_prob, proposal_prob) = self._guided_sample_trie(
                self.root, context, verbosity=verbosity, **kwargs
            )
        if compare_time:
            self.timer.compare()
        return (path, llm_prob, guide_prob, proposal_prob)

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

        return cpy

    def _guided_sample_trie(
        self, root, context, draw=sample_dict, verbosity=0
    ):

        curr = root
        path = []
        guide_prob = 1
        proposal_prob = 1
        exits = Float.chart()

        children = self.children
        mass = self.mass

        if verbosity > 1: print(colors.line(80))
        while True:

            children_curr = children[curr]
            mass_curr = mass[curr]

            p1 = Float.chart((a, mass[c]/mass_curr) for a, c in children_curr.items())

            p2 = self.guide.p_next(context + ''.join(path)).trim()

            if None in p1:
                exits[''.join(path)] = mass[children_curr[None]]
                if verbosity > 1: print(colors.blue % "ADDED EXIT", repr(''.join(path)), 'prob=', proposal_prob)

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

            if verbosity > 1: print(colors.orange % 'action', repr(a), 'context', repr(''.join(path)))

            path.append(a)

        # Sample the end-of-token marker in hindsight
        exits = exits.normalize()

        if verbosity > 1: print(colors.light.green % 'p exits:', exits)

        path = draw(exits)

        if verbosity > 1: print(colors.orange % 'picked exit', repr(path))

        proposal_prob *= exits[path]

        llm_prob = mass[self.word2leaf[path]]

        return (path, llm_prob, guide_prob, proposal_prob)

    def _enumerate_paths(self, context):
        # Used for debugging
        # MAKE SURE TO CALL proposal._update_trie(p_llm) BEFORE RUNNING

        curr = self.root
        children = self.children
        mass = self.mass
        paths = []
        exits = Float.chart()

        def _enum_paths(
            chars, trace, children_curr, mass_curr, proposal_prob, exits
        ):
            p1 = Float.chart(
                (a, mass[c]/mass_curr) for a, c in children_curr.items()
            )
            p2 = self.guide.p_next(context + ''.join(chars)).trim()

            if None in p1:
                exits[''.join(chars)] = mass[children_curr[None]]

            _q = (p1 * p2).trim()

            if not _q:
                # no more paths to explore
                exits = exits.normalize()
                these_paths = []
                for (token, exit_p) in exits.items():
                    new_trace = trace.copy()
                    new_trace.append({
                        'name' : 'exit',
                        'outcome' : token,
                        'prob' : exit_p,
                        'dist' : exits
                    })
                    these_paths.append({
                        'token' : token,
                        'proposal_prob' : proposal_prob * exit_p,
                        'trace' : new_trace
                    })
                return these_paths
            else:
                # keep exploring paths
                q = _q.normalize()
                for (a, q_prob) in q.items():
                    curr = children_curr[a]
                    new_chars = chars.copy()
                    new_chars.append(a)

                    new_exits = exits.copy()

                    new_trace = trace.copy()
                    new_trace.append({
                        'name' : f'char {len(new_chars)}',
                        'outcome' : a,
                        'prob' : q_prob,
                        'dist' : q
                    })
                    paths.extend(
                        _enum_paths(
                            chars=new_chars,
                            trace=new_trace,
                            children_curr=children[curr],
                            mass_curr=mass[curr],
                            proposal_prob=proposal_prob * q_prob,
                            exits=new_exits
                        )
                    )
                return []

        _enum_paths(
            chars=[],
            children_curr=children[curr],
            mass_curr=mass[curr],
            proposal_prob=1,
            exits=exits,
            trace=[]
        )

        return paths
