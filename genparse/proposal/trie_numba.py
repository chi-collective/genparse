import numba
import numpy as np

# from typing import Dict, List


class TokenCharacterTrie:

    __slots__ = (
        "root",
        "children",
        "mass",
        "word2leaf",
        "jump",
        "ordering",
        "old_eos",
        "new_eos",
        "token_id_to_leaf",
    )

    def __init__(self, words, encode, old_eos, new_eos):

        self.old_eos = old_eos
        self.new_eos = new_eos

        word2leaf = {}
        children = {}
        root = 0
        children = [{}]

        token_id_to_leaf = []

        for word in words:

            # coerce old eos to new eos
            _word = word
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

            token_id_to_leaf.append((encode[_word], last))

        self.token_id_to_leaf = token_id_to_leaf
        self.root = root
        self.children = children
        self.mass = np.zeros(len(children), dtype=np.float64)
        self.word2leaf = word2leaf
        self.jump = List(
            [np.array(sorted(x.values()), dtype=np.int32) for x in children]
        )
        self.ordering = np.array(list(self._order(self.root)), np.int32)

        # Renumber the states of the trie so that they are named by a contiguous
        # range of integers and those integers respect the are topologically
        # ordering of the trie topology.  This improves the efficiency of the
        # updating the trie as it improves memory locality.
        ordering = {}
        for i, x in enumerate(self._order_full(self.root)):
            ordering[x] = i
        self.rename(f=lambda x: ordering[x])

    # TODO: test this method!!!!!!
    def rename(self, f):
        N = len(self.mass)

        new_children = [{} for _ in range(N)]
        new_mass = np.zeros(N)

        nodes = range(N)

        for x in nodes:
            new_mass[f(x)] = self.mass[x]
            for letter, y in self.children[x].items():
                new_children[f(x)][letter] = f(y)

        self.root = f(self.root)
        self.children = new_children
        self.word2leaf = {w: f(x) for w, x in self.word2leaf.items()}

        self.token_id_to_leaf = np.array(
            [(i, f(x)) for i, x in self.token_id_to_leaf], dtype=np.int32
        )

        self.ordering = np.array([f(x) for x in self.ordering])
        self.jump = List(
            [np.array(sorted(x.values()), dtype=np.int32) for x in new_children]
        )

    def _update_trie(self, words):
        # convert llm.eos to guide.eos
        self.mass[self.word2leaf[self.new_eos]] = words[self.old_eos]

        _update_trie_numba(
            mass=self.mass,
            _p=words._p,
            token_id_to_leaf=self.token_id_to_leaf,
            jump=self.jump,
            ordering=self.ordering,
        )

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


from numba.typed import Dict, List


@numba.jit(nopython=True)
def _update_trie_numba(
    mass: numba.float64[:],
    _p: numba.float64[:],
    jump: List[numba.int32[:]],
    token_id_to_leaf: numba.int32[:, :],
    ordering: numba.int32[:],
):

    # update leaves
    M = token_id_to_leaf.shape[0]
    for k in range(M):
        i = token_id_to_leaf[k, 0]
        x = token_id_to_leaf[k, 1]
        mass[x] = _p[i]

    # update internal nodes
    N = ordering.shape[0]
    for i in range(N):
        node = ordering[i]
        total_mass = 0
        for child in jump[node]:
            total_mass += mass[child]
        mass[node] = total_mass
