import numpy as np
from arsenal.maths import sample_dict
#from genparse.inference import Node
from genparse import ERROR, Float
from arsenal import colors, timeit


class CharacterProposal:
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

    def __init__(self, llm, guide):
        self.llm = llm
        self.guide = guide

        self._init_trie(llm.V)

    def _update_trie(self, words):
        mass = self.mass; jump = self.jump

        # update leaves
        for word, leaf in self.word2leaf.items():
            mass[leaf] = words[word]
        # convert llm.eos to guide.eos
        mass[self.word2leaf[self.guide.eos]] = words[self.llm.eos]

        # update internal nodes (in bottom up order)
        for node in self.ordering:
            m = 0
            for child in jump[node]:
                m += mass[child]
            mass[node] = m

    def _init_trie(self, words):

        word2leaf = {}
        children = {}
        root = 0
        children = [{}]

        for word in words:

            # coerce llm.eos to guide.eos
            if word == self.llm.eos: word = self.guide.eos

            # Filter LLM tokens that are illegal under the cfg
            if not (set(word) <= self.guide.V): continue

            curr = root
            for letter in word:
                if letter not in children[curr]:
                    children[curr][letter] = len(children)
                    children.append({})
                curr = children[curr][letter]

            children[curr][None] = last = len(children)
            children.append({})
            word2leaf[word] = last

        self.root = root
        self.children = children
        self.mass = np.zeros(len(children))
        self.word2leaf = word2leaf
        self.jump = [tuple(sorted(x.values())) for x in children]
        self.ordering = list(self._order(self.root))

    def _order(self, node):
        "Topological ordering of nodes beneath `node`."
        mass = 0
        for a in self.children[node]:
            if a is None:
                pass
            else:
                yield from self._order(self.children[node][a])
        yield node

    def sample(self, prompt, max_tokens=float('inf'), verbosity=0, **kwargs):

        context = ''
        P = 1
        t = 0
        while True:

            self._update_trie(self.llm.p_next(prompt + context))

            token, p_token = self._guided_sample_trie(self.root, context, verbosity=verbosity, **kwargs)
            t += 1

            if t >= max_tokens:
                token = self.guide.eos
                p_token = 1

            P *= p_token

            if self.guide.eos == token:
                if verbosity > 0:
                    print()
                break

            if verbosity > 0:
                print(colors.cyan % token, end=colors.magenta % '|')

            context += token

        return (context, P)

    def _guided_sample_trie(self, root, context, draw=sample_dict, verbosity=0):

        curr = root
        path = []
        P = 1
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
                if verbosity > 1: print(colors.blue % "ADDED EXIT", repr(''.join(path)), 'prob=', P)

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
            P *= q[a]
            curr = children_curr[a]

            if verbosity > 1: print(colors.orange % 'action', repr(a), 'context', repr(''.join(path)))

            path.append(a)

        # Sample the end-of-token marker in hindsight
        exits = exits.normalize()

        if verbosity > 1: print(colors.light.green % 'p exits:', exits)

        path = draw(exits)

        if verbosity > 1: print(colors.orange % 'picked exit', repr(path))

        P *= exits[path]

        return (path, P)
