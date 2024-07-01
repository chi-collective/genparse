import dataclasses
import itertools
from collections.abc import Iterable, Sequence
from arsenal import iterview

State = int
Symbol = int
# A merge rule consists of (u, v, uv), where uv is the index of the symbol for
# the concatenation of u and v.
MergeRule = tuple[Symbol, Symbol, Symbol]


class TokenDFA:
    # The start state is always implicitly state 0.
    # Note that in this construction, all states are accept states, so there
    # is no explicit set of accept states.
    def __init__(self, base_alphabet):
        self.num_states = 1
        self.transitions = [{a: 0 for a in base_alphabet}]

    # Note: currently required for testing against other implementation
    def get_state_to(self, i, a):
        return self.transitions[i].get(a)

    @staticmethod
    def from_dictionary(
        base_alphabet: Iterable[Symbol], dictionary: Iterable[MergeRule]
    ) -> 'TokenDFA':
        dfa = TokenDFA.from_base_alphabet(base_alphabet)
        for rule in iterview(dictionary):
            dfa.merge_rule(rule)
            # TODO Add trimming? Find all states not reachable from start state.
            # This can probably be done during construction without needing to rescan
            # the automaton from scratch every time.
        return dfa

    @classmethod
    def from_base_alphabet(cls, base_alphabet: Iterable[Symbol]) -> 'TokenDFA':
        return cls(base_alphabet)

    def new_states(self, n: int) -> range:
        lo = self.num_states
        self.num_states += n
        new_states = range(lo, self.num_states)
        for _ in new_states:
            self.transitions.append({})
        return new_states

    #    def get_transitions(self) -> Iterable[tuple[State, Symbol, State]]:
    #        for state_from, transitions_from_state in self.transitions.items():
    #            for symbol, state_to in transitions_from_state.items():
    #                yield state_from, symbol, state_to

    def merge_rule(self, rule: MergeRule) -> None:
        # This implements Algorithm 2 of https://arxiv.org/pdf/2405.07671
        u, v, uv = rule
        transitions = self.transitions
        J = set()

        for i in range(self.num_states):
            j = transitions[i].get(u)
            if j is not None:
                k = transitions[j].get(v)
                if k is not None:
                    transitions[i][uv] = k
                    J.add(j)

        # print(f'{len(J)} / {self.num_states} = {len(J) / self.num_states}')

        if len(J) == 0:
            return

        fresh_J = dict(zip(J, self.new_states(len(J))))

        for j, fresh_j in fresh_J.items():
            for a, kʹ in transitions[j].items():
                if a == v:
                    continue
                if u == v and a == uv:
                    continue
                transitions[fresh_j][a] = kʹ

        for iʹ in range(self.num_states):
            j = transitions[iʹ].get(u)
            if j is not None:
                fresh_j = fresh_J.get(j)
                if fresh_j is not None:
                    transitions[iʹ][u] = fresh_j
