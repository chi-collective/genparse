import numpy as np


class TokenCharacterTrie:
    __slots__ = (
        'root',
        'children',
        'mass',
        'word2leaf',
        'jump',
        'ordering',
        'old_eos',
        'new_eos',
    )

    def __init__(self, words, old_eos, new_eos):
        self.old_eos = old_eos
        self.new_eos = new_eos

        word2leaf = {}
        children = {}
        root = 0
        children = [{}]

        for word in sorted(words):
            # coerce old eos to new eos
            if word == self.old_eos:
                word = self.new_eos

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
        self.jump = [list(sorted(x.values())) for x in children]
        self.ordering = list(self._order(self.root))

    def rename(self, f):
        N = len(self.mass)

        new_root = f(self.root)
        new_children = [{} for _ in range(N)]
        new_mass = np.zeros(N)

        nodes = range(N)

        for x in nodes:
            new_mass[f(x)] = self.mass[x]
            for letter, y in self.children[x].items():
                new_children[f(x)][letter] = f(y)

        new_jump = [None for _ in range(N)]
        for x in nodes:
            new_jump[f(x)] = tuple(sorted(f(y) for y in self.jump[x]))

        new_word2leaf = {w: f(x) for w, x in self.word2leaf.items()}
        new_ordering = [f(x) for x in self.ordering]

        self.root = new_root
        self.jump = new_jump
        self.children = new_children
        self.word2leaf = new_word2leaf
        self.ordering = new_ordering

    def _update_trie(self, words):
        self._update_leaves(words)
        self._update_internal()

    def _update_leaves(self, words):
        # update leaves
        mass = self.mass
        for word, leaf in self.word2leaf.items():
            mass[leaf] = words[word]
        # convert llm.eos to guide.eos
        mass[self.word2leaf[self.new_eos]] = words[self.old_eos]

    def _update_internal(self):
        mass = self.mass
        jump = self.jump
        # update internal nodes (in bottom up order)
        for node in self.ordering:
            m = 0
            for child in jump[node]:
                m += mass[child]
            mass[node] = m

    def _order(self, node):
        "Topological ordering of nodes beneath `node`."
        for a in self.children[node]:
            if a is None:
                pass
            else:
                yield from self._order(self.children[node][a])
        yield node

    def _order_full(self, node):
        "Topological ordering of nodes beneath `node`."
        for a in self.children[node]:
            yield from self._order_full(self.children[node][a])
        yield node
