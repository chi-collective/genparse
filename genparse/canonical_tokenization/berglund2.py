import numpy as np
import sys
from arsenal import iterview, Integerizer


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
    def from_dictionary(base_alphabet, dictionary):
        dfa = TokenDFA.from_base_alphabet(base_alphabet)
        for i, rule in enumerate(iterview(dictionary), start=1):
            dfa.merge_rule(rule)
            # TODO Add trimming? Find all states not reachable from start state.
            # This can probably be done during construction without needing to rescan
            # the automaton from scratch every time.
            if i % 5000 == 0:
                dfa.trim()
                print(dfa.num_states, 'states')
                print(sum(len(t) for t in dfa.transitions), 'arcs')

        print('finalize:')
        dfa.trim()
        print(dfa.num_states, 'states')
        print(sum(len(t) for t in dfa.transitions), 'arcs')

        return dfa

    @classmethod
    def from_base_alphabet(cls, base_alphabet):
        return cls(base_alphabet)

    def new_states(self, n):
        lo = self.num_states
        self.num_states += n
        new_states = range(lo, self.num_states)
        for _ in new_states:
            self.transitions.append({})
        return new_states

    def reachable(self):
        agenda = [0]
        chart = {0}
        transitions = self.transitions
        while agenda:
            i = agenda.pop()
            for j in transitions[i].values():
                if j not in chart:
                    chart.add(j)
                    agenda.append(j)
        return chart

    def trim(self):
        R = self.reachable()
        # renumber the states
        print(
            'reachable', len(R) / self.num_states, '=', len(R), 'out of', self.num_states
        )

        f = Integerizer()
        assert f(0) == 0  # ensure that 0 -> 0
        new = [{} for _ in R]
        old = self.transitions
        for i in sorted(R):
            for a, j in old[i].items():
                if j in R:
                    new[f(i)][a] = f(j)
        self.transitions = new
        self.num_states = len(R)

    def merge_rule(self, rule):
        # This implements Algorithm 2 of https://arxiv.org/pdf/2405.07671

        # A merge rule consists of (u, v, uv), where uv is the index of the
        # symbol for the concatenation of u and v.
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
