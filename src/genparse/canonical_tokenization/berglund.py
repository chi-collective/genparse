import dataclasses
import itertools
import sys
from collections.abc import Iterable, Sequence
from arsenal import iterview


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
        for rule in iterview(dictionary, transient=True):
            dfa.merge_rule(rule)
        return dfa

    @staticmethod
    def from_base_alphabet(base_alphabet: Iterable[Symbol]) -> 'TokenDFA':
        transitions_from_init = {a: 0 for a in base_alphabet}
        return TokenDFA(
            transitions=[transitions_from_init], in_degree=[len(transitions_from_init)]
        )

    @staticmethod
    def from_transitions(
        num_states: int, transitions: Iterable[tuple[State, Symbol, State]]
    ) -> 'TokenDFA':
        """Precondition: no transitions with the same source state and symbol."""
        dfa = TokenDFA.from_states(num_states)
        for q, a, r in transitions:
            dfa.add_transition(q, a, r)
        return dfa

    @staticmethod
    def from_states(num_states: int) -> 'TokenDFA':
        return TokenDFA(
            transitions=[{} for _ in range(num_states)], in_degree=[0] * num_states
        )

    def states(self) -> range:
        return range(len(self.transitions))

    def num_states(self) -> int:
        return len(self.transitions)

    def new_state(self) -> State:
        new_state = len(self.transitions)
        self.transitions.append({})
        self.in_degree.append(0)
        return new_state

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
        # First, renumber the states so that their int values are contiguous.
        new_state_ints = {}
        for state, transitions_from_state in enumerate(self.transitions):
            if transitions_from_state is not _NO_TRANSITIONS:
                new_state_ints[state] = len(new_state_ints)
        # Now, generate the transitions with the renumbered states.
        for state_from, transitions_from_state in enumerate(self.transitions):
            for symbol, state_to in transitions_from_state.items():
                yield new_state_ints[state_from], symbol, new_state_ints[state_to]

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
        """Precondition: The initial DFA must have had targetc at most 1
        (which is the case for the base alphabet DFA). Otherwise, the result
        of this method might be incorrect."""
        # This implements Algorithm 2 of https://arxiv.org/pdf/2405.07671
        u, v, uv = rule
        # The general form of Berglund's algorithm uses a *set* S2 of states.
        # However, the maximum size of S2 depends on properties of the base
        # alphabet DFA, and in our case it's guaranteed to be of size at most
        # 1. So we only need to remember one state s2.
        s2 = None
        for s1 in self.states():
            maybe_s2 = self.get_state_to(s1, u)
            if maybe_s2 is not None:
                s3 = self.get_state_to(maybe_s2, v)
                if s3 is not None:
                    self.add_transition(s1, uv, s3)
                    s2 = maybe_s2
        # If S2 is empty, the rest of this algorithm is a no-op. Stop early to
        # save time.
        if s2 is None:
            return
        fresh_s2 = self.new_state()
        excluded = [v]
        if u == v:
            excluded.append(uv)
        for alpha, state_to in self.get_transitions_from_state(s2):
            if alpha not in excluded:
                self.add_transition(fresh_s2, alpha, state_to)
        for q in self.states():
            if self.get_state_to(q, u) == s2:
                self.reset_transition(q, u, fresh_s2)
                self.check_state_for_removal(s2)
