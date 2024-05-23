from arsenal.maths import sample_dict
#from genparse.inference import Node
from genparse import ERROR, Float
from arsenal import colors, timeit

class Node:
    __slots__ = ('mass', 'children')

    def __init__(self, mass, children=None):
        self.mass = mass
        self.children = children

    def p_next(self):
        return Float.chart((a, c.mass/self.mass) for a, c in self.children.items())


class TokenTrieApproximation:
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

        self.llm_eos = llm.eos
        (self.root, self.word2leaf) = self.make_skeleton(llm.V)

        self.ordering = list(self._order(self.root))

    def about(self):
        print('internal trie nodes:', len(self.ordering))
        print('leaves:', len(self.word2leaf))
        print('ratio:', len(self.ordering) / len(self.word2leaf))
        print('avg branching:', sum(len(x.children) for x in self.ordering)/len(self.ordering))
        print('max branching:', max(len(x.children) for x in self.ordering if x != self.root))
        print('longest word:', max(len(w) for w in self.word2leaf))

    def _update_trie(self, words):

        words[self.guide.eos] = words[self.llm_eos]

        for word, leaf in self.word2leaf.items():
            leaf.mass = words[word]

        self._propagate_mass(self.root)
#        self._propagate_mass_loop()

        return self.root

#    def _propagate_mass_loop(self):
#        for node in self.ordering:
#            mass = 0
#            for child in node.children.values():
#                mass += child.mass
#            node.mass = mass

    def _order(self, node):
        mass = 0
        for a in node.children:
            if a is None:
                pass
            else:
                yield from self._order(node.children[a])
        yield node

#    def show(self, **kwargs):
#        _kwargs = dict(fmt_edge = lambda x,a,y: f'{a}/{y.mass/x.mass:.2g}',
#                       fmt_node=lambda x: f'{x.mass:.2g}')
#        _kwargs.update(**kwargs)
#        return self.root.graphviz(**_kwargs)

    def make_skeleton(self, words):
        # For efficiency, we use trie skeleton and propagate values on it.
        root = Node(mass=None, children={})

        word2leaf = {}
        for word in sorted(words):
            if word == self.llm_eos: word = self.guide.eos

            # Filter LLM tokens that are illegal under the cfg
            if not (set(word) <= self.guide.V): continue

            curr = root
            for letter in word:
                if letter not in curr.children:
                    curr.children[letter] = Node(mass=None, children={})
                curr = curr.children[letter]
            curr.children[None] = last = Node(mass=None, children=None)
            word2leaf[word] = last

        return (root, word2leaf)

    def _propagate_mass(self, node):
        mass = 0
        for a, child in node.children.items():
            if a is None:
                mass += child.mass
            else:
                mass += self._propagate_mass(child)
        node.mass = mass
        return mass

    def sample(self, prompt, max_tokens=float('inf'), prob=False, verbosity=0, **kwargs):

        context = ''
        P = 1
        t = 0
        while True:

            p1 = self.llm.p_next(prompt + context)

            # Convert a flat probability distribution over tokens into a trie-structured
            # distribution over characters and an end-of-token marker (`None`).
            p1_trie_root = self._update_trie(p1)
            ys, P_ys = self._guided_sample_trie(p1_trie_root, context, verbosity=verbosity, **kwargs)
            t += 1

            if t >= max_tokens:
                ys = self.guide.eos
                P_ys = 1

            P *= P_ys

            if ERROR in ys:
                print(colors.light.red % 'error case')
                context += ys
                break

            if self.guide.eos == ys:
                if verbosity > 0:
                    print()
                break

            y = ''.join(ys)

            if verbosity > 0:
                print(colors.cyan % y, end=colors.magenta % '|')

            context += y

        return (context, P) if prob else context

    def _guided_sample_trie(self, root, context, draw=sample_dict, verbosity=0):

        curr = root
        path = []
        P = 1
        exits = Float.chart()

        if verbosity > 1: print(colors.line(80))
        while True:

            p1 = curr.p_next()
            p2 = self.guide.p_next(context + ''.join(path)).trim()

            if None in p1:
                exits[''.join(path)] = curr.children[None].mass
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
            curr = curr.children[a]

            if verbosity > 1: print(colors.orange % 'action', repr(a), 'context', repr(''.join(path)))

            path.append(a)

        # Sample the end-of-token marker in hindsight
        exits = exits.normalize()

        if verbosity > 1: print(colors.light.green % 'p exits:', exits)

        path = draw(exits)

        if verbosity > 1: print(colors.orange % 'picked exit', repr(path))

        P *= exits[path]

        return (path, P)
