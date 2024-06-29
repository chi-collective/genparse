from genparse.canonical_tokenization.berglund import TokenDFA


def are_isomorphic(dfa1, dfa2, vocab_as_int):
    if dfa1.num_states != dfa2.num_states:
        return False
    agenda = [0]
    state_mapping = {0: 0}
    while agenda:
        q1 = agenda.pop()
        for a in vocab_as_int:
            r1 = dfa1.get_state_to(q1, a)
            r2 = dfa2.get_state_to(state_mapping[q1], a)
            if r1 is None:
                if r2 is not None:
                    return False
            else:
                if r2 is None:
                    return False
                else:
                    if r1 in state_mapping:
                        if r2 != state_mapping[r1]:
                            return False
                    else:
                        state_mapping[r1] = r2
                        agenda.append(r1)
    return True


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
        M = TokenDFA(num_states=num_states, transitions={})
        for q, a, r in transitions:
            M.set_transition(q, str_to_int[a], r)
        return M

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
