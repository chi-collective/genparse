import collections
import dataclasses
import itertools
from collections.abc import Iterable, Sequence

State = int
Symbol = int
MergeRule = tuple[Symbol, Symbol]


@dataclasses.dataclass
class DFA:
    num_states: int
    alphabet_size: int
    transitions: dict[State, collections.OrderedDict[Symbol, State]]
    # Note that in this construction, all states are accept states, so there
    # is no explicit set of accept states.

    def new_states(self, n: int) -> range:
        lo = self.num_states
        self.num_states += n
        return range(lo, self.num_states)

    def new_symbol(self) -> Symbol:
        a = self.alphabet_size
        self.alphabet_size += 1
        return a

    def get_state_to(self, state_from: State, symbol: Symbol) -> State | None:
        d = self.transitions.get(state_from)
        if d is not None:
            return d.get(symbol)

    def get_transitions_from_state(self, state_from: State) -> Iterable[Symbol, State]:
        d = self.transitions.get(state_from)
        if d is not None:
            return d.items()
        else:
            return ()

    def set_transition(self, state_from: State, symbol: Symbol, state_to: State) -> None:
        if state_from not in self.transitions:
            self.transitions[state_from] = collections.OrderedDict()
        self.transitions[state_from][symbol] = state_to


def construct_base_token_dfa(alphabet_size: int) -> DFA:
    return DFA(
        num_states=1,
        alphabet_size=alphabet_size,
        transitions={0: collections.OrderedDict.fromkeys(range(alphabet_size), 0)},
    )


def merge_rule_into_token_dfa(dfa: DFA, rule: MergeRule) -> None:
    # This implements Algorithm 2 from https://arxiv.org/pdf/2405.07671
    u, v = rule
    uv = dfa.new_symbol()
    # Use OrderedDict to ensure deterministic iteration order.
    S2 = collections.OrderedDict()
    for s1 in range(dfa.num_states):
        s2 = dfa.get_state_to(s1, u)
        if s2 is not None:
            s3 = dfa.get_state_to(s2, v)
            if s3 is not None:
                dfa.set_transition(s1, uv, s3)
                S2[s2] = True
    fresh = dfa.new_states(len(S2))
    excluded = [v]
    if u == v:
        excluded.append(uv)
    for s2, fresh_s2 in zip(S2, fresh):
        for alpha, state_to in dfa.get_transitions_from_state(s2):
            if alpha not in excluded:
                dfa.set_transition(fresh_s2, alpha, state_to)
    state_to_fresh = dict(zip(S2, fresh))
    for q in range(dfa.num_states):
        state_to = dfa.get_state_to(q, u)
        if state_to is not None:
            fresh_state_to = state_to_fresh.get(state_to)
            if fresh_state_to is not None:
                dfa.set_transition(q, u, fresh_state_to)


def construct_token_dfa(alphabet_size: int, dictionary: Iterable[MergeRule]) -> DFA:
    dfa = construct_base_token_dfa(alphabet_size)
    for rule in dictionary:
        merge_rule_into_token_dfa(dfa, rule)
        # TODO Add trimming? Find all states not reachable from start state.
        # This can probably be done during construction without needing to rescan
        # the automaton from scratch every time.
    return dfa


def get_int_mapping(
    alphabet: Iterable[str], dictionary: Iterable[tuple[str, str]]
) -> Iterable[str]:
    return itertools.chain(alphabet, (u + v for u, v in dictionary))
