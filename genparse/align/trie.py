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

    def sample(self, prompt, max_tokens, prob=False, **kwargs):

        verbosity = kwargs.get('verbosity', 0)

        context = ''
        P = 1
        for _ in range(max_tokens):

            p1 = self.llm.p_next(prompt + context).normalize()
            p1_trie = NextTokenTrie(p1, self.llm.eos)
            ys, P_ys = self._guided_sample_trie(p1_trie.root, context, **kwargs)

            P *= P_ys

            if ERROR in ys:
                print(colors.light.red % 'error case')
                context += y
                break

            if EOS in ys:
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

        while True:

            p1 = curr.p_next()
            p2 = self.guide.p_next(context + ''.join(path)).trim()
            p2 = Float.chart({x: 1 for x in p2})
#            print(p2)

            _q = (p1 * p2).trim()

            if not _q:
                break

            q = _q.normalize()

            if verbosity > 1:
                print(colors.yellow % 'partial token=', repr(''.join(path)))
                print(colors.yellow % 'calling context=', repr(''.join(context)))
                if not q or verbosity > 2:
                    print('llm=',p1)
                    print('guide=',p2)
                print('q=',q)

            a = draw(q)
            P *= q[a]
            P1 *= p1[a]
            P2 *= p2[a]

            curr = curr.children[a]

#            print(colors.orange % 'action', repr(a), 'context', repr(''.join(path)))

            path.append(a)

            # XXX: Warning the BPE vocabulary is not prefix closed!
            # XXX: It's possible that we should pick the stop by *only* the llm probability
            if None in p1:
                exits[''.join(path)] = P
#            print("ADDED EXIT", repr(''.join(path)), 'prob=', P)


        # Sample the end-of-token marker in hindsight
#        print('exits (unnormalized):', exits)
        exits = exits.normalize()

#        if len(exits) == 0:
#            self._guided_sample_trie(root, context, draw=draw, verbosity=verbosity)

#        print(colors.light.green % 'exits:', exits)

        path = draw(exits)

#        print(colors.orange % 'picked exit', repr(path))

        P *= exits[path]

        return (path, P)


def test_llm_trie_approximation():

    from genparse.util import LarkStuff
    from genparse.inference import TraceSWOR
    from genparse import CFGLM, locally_normalize
    from genparse.lm import GreedilyTokenizedLLM

    import numpy as np
    np.random.seed(0)

    pcfg = CFGLM(locally_normalize(LarkStuff(r"""

    start: /[ ]*Tim(othy)?[ ](Fabbri[ ])?Vieira\./

    """).char_cfg(.99), tol=1e-100))

    prompt = 'Hello my name is'
    llm = GreedilyTokenizedLLM("gpt2")

    llm.sample('My name is', verbose=1, max_tokens=10)

    token_trie_approx = TokenTrieApproximation(llm, pcfg)
    tracer = TraceSWOR()
    for _ in range(10):
        with tracer:
            print('----------------------------------')
            with timeit('complete sample'):
                ys = token_trie_approx.sample(prompt, max_tokens=50,
                                              draw=tracer,
                                              verbosity=1)
            print(colors.light.yellow % 'sample:', ys)


def test_the_linguistic_said():

    from genparse.util import LarkStuff
    from genparse.inference import TraceSWOR
    from genparse import CFGLM, locally_normalize
    from genparse.lm import GreedilyTokenizedLLM

    import numpy as np
    np.random.seed(0)

    pcfg = CFGLM(locally_normalize(LarkStuff(r"""

    start: /Noam[ ]Chomsky[ ]famously[ ]wrote,[ ]"/ expr /\."/

    expr: /[A-Za-z0-9,; ]+/
//    expr: /[Tt]ime[ ]flies[ ]like[ ]an[ ]arrow/
//        | /[iI][ ]like[ ]to[ ]dance/
//        | /Colorless[ ]green[ ]ideas[ ]sleep[ ]furiously/

    """).char_cfg(.99), tol=1e-100))

    print(''.join(pcfg.sample()))

    prompt = ' '
    llm = GreedilyTokenizedLLM("gpt2")

    W = Float.chart()

    token_trie_approx = TokenTrieApproximation(llm, pcfg)
#    tracer = TraceSWOR()
    tracer = sample_dict
    for _ in range(10):
#        with tracer:
            print('----------------------------------')
            with timeit('complete sample'):
                ys, q = token_trie_approx.sample(prompt, max_tokens=50,
                                                 draw=tracer, prob=True,
                                                 verbosity=1)
            score = llm(ys) * pcfg(ys + EOS)
            W[ys] += score / q

            print(q)

            print(colors.light.yellow % 'sample:', ys)

            print(W.normalize())


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
