from functools import lru_cache
from collections import defaultdict


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


# TODO: unused; untested
def topologically_order_merge_list(vocab, merge_list):
    # The set of composite tokens are those that are defined by concatenation of
    # at least one pair of tokens by a merge rule.

    # map from a token string to its possible splits into sub-tokens
    tmp = defaultdict(list)
    for x, y in merge_list:
        tmp[x + y].append((x, y))
    composite_tokens = set(tmp)

    # the base alphabet is any token in the vocabulary that is not a composite
    # token.
    base_alphabet = set()
    for u, v in merge_list:
        for x in (u, v):
            if x not in composite_tokens:
                base_alphabet.add(x)
    for x in vocab:
        if x not in composite_tokens:
            base_alphabet.add(x)

    # TODO: untested
    assert base_alphabet == vocab - composite_tokens

    @lru_cache
    def height(x):
        # The height of x's tallest derivation."
        if x in base_alphabet:
            return 1
        else:
            return 1 + max(max(height(u), height(v)) for u, v in tmp[x])

    # enforce the topological ordering
    return sorted(merge_list, key=lambda uv: height(uv[0] + uv[1]))
