import dataclasses
import itertools
import sys
from collections.abc import Iterable, Sequence


State = int
Symbol = int
# A merge rule consists of (u, v, uv), where uv is the index of the symbol for
# the concatenation of u and v.
MergeRule = tuple[Symbol, Symbol, Symbol]

_NO_TRANSITIONS = {}


@dataclasses.dataclass
class TokenDFA:
    # The start state is always implicitly state 0.
    # Note that in this construction, all states are accept states, so there
    # is no explicit set of accept states.
    transitions: list[dict[Symbol, State]]
    in_degree: list[int]

    @staticmethod
    def from_dictionary(
        base_alphabet: Iterable[Symbol], dictionary: Iterable[MergeRule]
    ) -> 'TokenDFA':
        dfa = TokenDFA.from_base_alphabet(base_alphabet)
        for rule in dictionary:
            dfa.merge_rule(rule)
            # TODO Add trimming? Find all states not reachable from start state.
            # This can probably be done during construction without needing to rescan
            # the automaton from scratch every time.
        return dfa

    @staticmethod
    def from_base_alphabet(base_alphabet: Iterable[Symbol]) -> 'TokenDFA':
        transitions_from_init = {a: 0 for a in base_alphabet}
        return TokenDFA(
            transitions=[transitions_from_init], in_degree=[len(transitions_from_init)]
        )

    def states(self) -> range:
        return range(len(self.transitions))

    def new_states(self, n: int) -> range:
        prev_len = len(self.transitions)
        for _ in range(n):
            self.transitions.append({})
            self.in_degree.append(0)
        return range(prev_len, len(self.transitions))

    def get_state_to(self, state_from: State, symbol: Symbol) -> State | None:
        return self.transitions[state_from].get(symbol)

    def get_transitions_from_state(
        self, state_from: State
    ) -> Iterable[tuple[Symbol, State]]:
        return self.transitions[state_from].items()

    def get_transitions(self) -> Iterable[tuple[State, Symbol, State]]:
        for state_from, transitions_from_state in enumerate(self.transitions):
            for symbol, state_to in transitions_from_state.items():
                yield state_from, symbol, state_to

    def add_transition(self, state_from: State, symbol: Symbol, state_to: State) -> None:
        """Precondition: state_from does not already have a transition on symbol."""
        self.transitions[state_from][symbol] = state_to
        self.in_degree[state_to] += 1

    def reset_transition(
        self, state_from: State, symbol: Symbol, state_to: State
    ) -> None:
        """Precondition: state_from already has a transition on symbol."""
        self.in_degree[self.transitions[state_from][symbol]] -= 1
        self.add_transition(state_from, symbol, state_to)

    def check_state_for_removal(self, state: State) -> None:
        if self.in_degree[state] == 0:
            agenda = [state]
            discovered = set(agenda)
            # NOTE: No states are ever removed recursively in practice. Is it
            # even possible for this to be necessary?
            while agenda:
                state = agenda.pop()
                for state_to in self.transitions[state].values():
                    self.in_degree[state_to] -= 1
                    if self.in_degree[state_to] == 0 and state_to not in discovered:
                        discovered.add(state_to)
                        agenda.append(state_to)
                self.transitions[state] = _NO_TRANSITIONS

    def merge_rule(self, rule: MergeRule) -> None:
        # This implements Algorithm 2 of https://arxiv.org/pdf/2405.07671
        u, v, uv = rule
        # Use dict to ensure deterministic iteration order.
        S2 = {}
        for s1 in self.states():
            s2 = self.get_state_to(s1, u)
            if s2 is not None:
                s3 = self.get_state_to(s2, v)
                if s3 is not None:
                    self.add_transition(s1, uv, s3)
                    S2[s2] = True
        # If S2 is empty, the rest of this algorithm is a no-op. Stop early to
        # save time.
        if not S2:
            return
        fresh = self.new_states(len(S2))
        excluded = [v]
        if u == v:
            excluded.append(uv)
        for s2, fresh_s2 in zip(S2, fresh):
            for alpha, state_to in self.get_transitions_from_state(s2):
                if alpha not in excluded:
                    self.add_transition(fresh_s2, alpha, state_to)
        state_to_fresh = dict(zip(S2, fresh))
        for q in self.states():
            state_to = self.get_state_to(q, u)
            if state_to is not None:
                fresh_state_to = state_to_fresh.get(state_to)
                if fresh_state_to is not None:
                    self.reset_transition(q, u, fresh_state_to)
                    self.check_state_for_removal(state_to)
