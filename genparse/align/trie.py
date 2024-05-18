from arsenal.maths import sample_dict
from genparse.inference import Node
from genparse import EOS, ERROR, Float
from arsenal import colors, timeit


SKELETON = None

class NextTokenTrie:
    """
    Convert a flat probability distribution over strings into a trie-structured
    distribution over characters and an end-of-token marker (`None`).
    """

    def __init__(self, p_next_token, llm_eos):
        self.p_next_token = p_next_token
        self.llm_eos = llm_eos
        self.root = self._make_trie(p_next_token)

    def _make_trie(self, words):

        global SKELETON
        if SKELETON is None:
            self.make_skeleton(words)

        (root, word2leaf) = SKELETON

        for word in words:
            mass = words[word]
            if word == self.llm_eos:
                word = EOS
            leaf = word2leaf[word]
            leaf.mass = leaf._mass = mass

        # push mass up from the leaves; uses DFS traversal
        self._propagate_mass(root)

        return root

    def make_skeleton(self, words):
        # TODO: For efficiency, we should use trie skeleton and propagate the values
        #
        # build the probability tree; assigning mass to the leaves
        global SKELETON
        root = Node(mass=None, parent=None, children={})

        word2leaf = {}
        for word in sorted(words):
            mass = words[word]
            # rename the EOS token so that it matches the guide's EOS token.
            if word == self.llm_eos: word = EOS     # TODO: harded coded guide's EOS token
            curr = root
            for letter in list(word) + [None]:
                if letter not in curr.children:
                    curr.children[letter] = Node(mass=None, parent=curr, children={})
                curr = curr.children[letter]
            curr.children = None
            curr.mass = curr._mass = mass
            word2leaf[word] = curr

        SKELETON = (root, word2leaf)
        return SKELETON

    def _propagate_mass(self, node):
        mass = 0
        for a in node.children:
            if a is None:
                mass += node.children[a].mass
            else:
                mass += self._propagate_mass(node.children[a])
        node.mass = node._mass = mass
        return mass

    def show(self, **kwargs):
        _kwargs = dict(fmt_edge = lambda x,a,y: f'{a}/{y.mass/x.mass:.2g}',
                       fmt_node=lambda x: f'{x.mass:.2g}')
        _kwargs.update(**kwargs)
        return self.root.graphviz(**_kwargs)


class TokenTrieApproximation:

    def __init__(self, llm, guide):
        self.llm = llm
        self.guide = guide

    def sample(self, prompt, max_tokens=float('inf'), prob=False, **kwargs):

        verbosity = kwargs.get('verbosity', 0)

        context = ''
        P = 1
        t = 0
        while True:

            p1 = self.llm.p_next(prompt + context).normalize()
            p1_trie = NextTokenTrie(p1, self.llm.eos)
            ys, P_ys = self._guided_sample_trie(p1_trie.root, context, **kwargs)
            t += 1

            if t >= max_tokens:
                ys = EOS
                P_ys = 1

            P *= P_ys

            if ERROR in ys:
                print(colors.light.red % 'error case')
                context += y
                break

            if EOS == ys:
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
        P1 = 1
        P2 = 1

        exits = Float.chart()

        if verbosity > 1: print(colors.line(80))
        while True:

            p1 = curr.p_next()
            p2 = self.guide.p_next(context + ''.join(path)).trim()

            # booleanized
            #p2 = Float.chart({x: 1 for x in p2})
            #print(p2)

            if None in p1:
                exits[''.join(path)] = curr.children[None]._mass
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
            P1 *= p1[a]
            P2 *= p2[a]

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
