from genparse.canonical_tokenization.berglund import TokenDFA
from genparse.canonical_tokenization.util import are_isomorphic


def test_example_2():
    # This tests Example 2 of https://arxiv.org/abs/2405.07671

    alphabet = ['a', 'b']
    dictionary = [('a', 'a'), ('b', 'a')]

    int_to_str = sorted(set(alphabet) | {u + v for u, v in dictionary})
    str_to_int = {s: i for i, s in enumerate(int_to_str)}

    vocab_as_int = range(len(int_to_str))
    alphabet_as_int = [str_to_int[a] for a in alphabet]
    dictionary_as_int = [
        (str_to_int[u], str_to_int[v], str_to_int[u + v]) for u, v in dictionary
    ]

    def construct_dfa(num_states, transitions):
        return TokenDFA.from_transitions(
            num_states, ((q, str_to_int[a], r) for q, a, r in transitions)
        )

    M = TokenDFA.from_base_alphabet(alphabet_as_int)
    N = construct_dfa(num_states=1, transitions=[(0, 'a', 0), (0, 'b', 0)])
    assert are_isomorphic(M, N, vocab_as_int)

    M.merge_rule(dictionary_as_int[0])
    N = construct_dfa(
        num_states=2,
        transitions=[(0, 'aa', 0), (0, 'b', 0), (0, 'a', 1), (1, 'b', 0)],
    )
    assert are_isomorphic(M, N, vocab_as_int)

    M.merge_rule(dictionary_as_int[1])
    N = construct_dfa(
        num_states=3,
        transitions=[
            (0, 'aa', 0),
            (0, 'b', 2),
            (0, 'a', 1),
            (0, 'ba', 1),
            (1, 'ba', 1),
            (1, 'b', 2),
            (2, 'b', 2),
            (2, 'aa', 0),
            (2, 'ba', 1),
        ],
    )
    assert are_isomorphic(M, N, vocab_as_int)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
