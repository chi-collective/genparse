class NextTokenTrie:
    """
    Convert a flat probability distribution over strings into a trie-structured
    distribution over characters and an end-of-token marker (`None`).
    """

    def __init__(self, p_next_token):
        self.p_next_token = p_next_token
        self.root = self._make_trie(p_next_token)

    def _make_trie(self, words):
        root = Node(mass=None, parent=None, children={})

        # build the probability tree; assigning mass to the leaves
        for word, mass in words.items():

            # rename the EOS token so that it matches the CFGLM's EOS token.
            # TODO: we need to ensure that we are using the correct EOS token,
            # this is hard coded for gpt2
            if word == '<|endoftext|>':
                word = EOS
                #print('EOS:', mass)

            curr = root
            for letter in list(word) + [None]:
                if letter not in curr.children:
                    curr.children[letter] = Node(mass=None, parent=curr, children={})
                curr = curr.children[letter]
            curr.children = None
            curr.mass = curr._mass = mass

        # push mass up from the leaves; uses DFS traversal
        self._propagate_mass(root)

        return root

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

    def sample(self, prompt, max_tokens, **kwargs):

        context = ''
        for _ in range(max_tokens):

            p1 = self.llm.p_next(prompt + context).normalize()
            p1_trie = NextTokenTrie(p1)
            ys = self._guided_sample_trie(p1_trie.root, context, **kwargs)[1]

            if ERROR in ys:
                print('error case')
                context += y
                break

            if EOS in ys:
                break

            if not ys:
                break

            y = ''.join(ys)

            context += y

        return context

    def _guided_sample_trie(self, root, context, draw=sample_dict, verbosity=0):

        curr = root
        path = []
        P = 1

        if verbosity > 0:
            print(colors.cyan % 'context=', repr(context))

        while True:
            # Character and end-of-token (`None`) probabilities from trie
            p1 = curr.p_next().normalize()

            p2 = self.guide.p_next(context + ''.join(path)).normalize()

            # Determining the weight of the end-of-token marker is tricky as it
            # is not part of the character-level model; Here we heuristically
            # set it to 1.
            p2[None] = 1

            _q = (p1 * p2).trim()

            if not _q:
                print('dead end')
                a = ERROR
                P *= 0
                path.append(a)
                break

            q = _q.normalize()

            if verbosity > 1:
                print(colors.yellow % 'partial token=', repr(''.join(path)))
                print('llm=',p1)
                print('guide=',p2)
                print('q=',q)


            a = draw(q)
            P *= q[a]
            curr = curr.children[a]
            if a is None: break
            if a == EOS: break
            path.append(a)

        return (P, path, curr)
        

def test_llm_trie_approximation():

    from genparse.util import LarkStuff
    from genparse.inference import TraceSWOR
    from genparse import CFGLM, locally_normalize
    from genparse.lm import GreedilyTokenizedLLM

    pcfg = CFGLM(locally_normalize(LarkStuff(r"""

    start: /[ ]*Tim(othy)?[ ](Fabbri[ ])?Vieira\./

    """).char_cfg(.99), tol=1e-100))

    prompt = 'Hello my name is'
    llm = GreedilyTokenizedLLM("gpt2")

    token_trie_approx = TokenTrieApproximation(llm, pcfg)
    tracer = TraceSWOR()
    for _ in range(1):
        with tracer:
            print('----------------------------------')
            print(tracer.root.mass)
            ys = token_trie_approx.sample(prompt, max_tokens=50, draw=tracer, verbosity=1)
            print(ys)


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())

