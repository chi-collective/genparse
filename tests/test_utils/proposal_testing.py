###########################
# Trace enumeration utils

from genparse import Float
import copy


class Trace:
    def __init__(self):
        self.score = 1
        self.choices = []
        self.path = None
        self.token = None
        self.weight = None

    def record(self, name, outcome, dist):
        self.score *= dist[outcome]
        self.choices.append(
            {'name': name, 'outcome': outcome, 'p': dist[outcome], 'dist': dist}
        )

    def __repr__(self):
        return f'{self.path}→`{self.token}` : {self.weight}'


def enumerate_traces(proposal, prompt, context):
    p_llm = proposal.llm.p_next(prompt + context)
    proposal._update_trie(p_llm)

    curr = proposal.root
    children = proposal.children
    mass = proposal.mass

    def _enum_traces(chars, trace, children_curr, mass_curr, cfg_p, inc_p, weights):
        p1 = Float.chart((a, mass[c] / mass_curr) for a, c in children_curr.items())
        p2 = proposal.guide.p_next(context + ''.join(chars)).trim()

        if None in p1:
            weights[''.join(chars)] = (mass[children_curr[None]] * cfg_p) / inc_p

        _q = (p1 * p2).trim()

        traces = []
        if not _q:
            normalized_weights = weights.normalize()
            for token in normalized_weights.keys():
                new_trace = copy.deepcopy(trace)
                new_trace.record('exit', token, normalized_weights)

                new_trace.token = token
                new_trace.weight = weights.sum()
                new_trace.path = '→'.join(chars)

                traces.append(new_trace)
        else:
            q = _q.normalize()
            for a, q_ in q.items():
                curr = children_curr[a]
                new_chars = chars.copy()
                new_chars.append(a)

                new_trace = copy.deepcopy(trace)
                new_trace.record(f'char {len(new_chars)}', a, q)

                traces.extend(
                    _enum_traces(
                        chars=new_chars,
                        trace=new_trace,
                        children_curr=children[curr],
                        mass_curr=mass[curr],
                        inc_p=inc_p * q_,
                        cfg_p=cfg_p * p2[a],
                        weights=weights.copy(),
                    )
                )

        return traces

    return _enum_traces(
        chars=[],
        trace=Trace(),
        children_curr=children[curr],
        mass_curr=mass[curr],
        cfg_p=1,
        inc_p=1,
        weights=Float.chart(),
    )


def enumerate_traces_uncorrected(proposal, prompt, context):
    p_llm = proposal.llm.p_next(prompt + context)
    proposal._update_trie(p_llm)

    curr = proposal.root
    children = proposal.children
    mass = proposal.mass

    def _enum_traces(chars, trace, children_curr, mass_curr, cfg_p, exits):
        p1 = Float.chart((a, mass[c] / mass_curr) for a, c in children_curr.items())
        p2 = proposal.guide.p_next(context + ''.join(chars)).trim()

        if None in p1:
            exits[''.join(chars)] = mass[children_curr[None]]

        _q = (p1 * p2).trim()

        traces = []
        if not _q:
            # no more paths to explore
            exits_norm = exits.normalize()
            for token, exit_p in exits_norm.items():
                new_trace = copy.deepcopy(trace)
                new_trace.record('exit', token, exits_norm)

                new_trace.token = token
                new_trace.weight = (
                    cfg_p * mass[proposal.word2leaf[token]] / new_trace.score
                )
                new_trace.path = '→'.join(chars)

                traces.append(new_trace)
        else:
            q = _q.normalize()
            for a, q_ in q.items():
                curr = children_curr[a]
                new_chars = chars.copy()
                new_chars.append(a)

                new_trace = copy.deepcopy(trace)
                new_trace.record(f'char {len(new_chars)}', a, q)

                traces.extend(
                    _enum_traces(
                        chars=new_chars,
                        trace=new_trace,
                        children_curr=children[curr],
                        cfg_p=cfg_p * p2[a],
                        mass_curr=mass[curr],
                        exits=exits.copy(),
                    )
                )
        return traces

    return _enum_traces(
        chars=[],
        trace=Trace(),
        children_curr=children[curr],
        mass_curr=mass[curr],
        cfg_p=1,
        exits=Float.chart(),
    )


def enumerate_target(proposal, prompt, context):
    p_next = Float.chart()
    for token in proposal.llm.V:
        cfg_prob = 1
        for i in range(0, len(token)):
            cfg_prob *= proposal.guide.p_next(context + token[:i])[token[i]]
        p_next[token] = cfg_prob * proposal.llm.p_next(prompt + context)[token]
    return p_next


def make_character_proposal(V, grammar, uniform=False):
    from genparse.lm import MockLLM
    from genparse.proposal import CharacterProposal
    from genparse.cfglm import EarleyBoolMaskCFGLM
    from genparse.util import LarkStuff
    from arsenal.maths import random_dist

    llm = MockLLM(V=V, eos='▪', _p=None if uniform else random_dist(len(V)))
    guide = EarleyBoolMaskCFGLM(LarkStuff(grammar).char_cfg(0.99, ignore='[ ]?'))

    return CharacterProposal(llm=llm, guide=guide)
