import asyncio
from arsenal.maths import random_dist, assert_equal

from genparse.trace import TraceSWOR
from genparse import Float
from genparse.lm import MockLLM, LM
from genparse.proposal import TokenProposal, CharacterProposal
from genparse.cfglm import EarleyBoolMaskCFGLM
from genparse.util import LarkStuff


def _make_guide(guide_spec):
    if isinstance(guide_spec, str):
        return EarleyBoolMaskCFGLM(LarkStuff(guide_spec).char_cfg(0.99, ignore='[ ]?'))
    elif isinstance(guide_spec, LM):
        return guide_spec
    else:
        raise ValueError('Unknown guide specification')  # pragma: no cover


def _make_mock_llm(V, uniform):
    return MockLLM(V=V, eos='▪', _p=None if uniform else random_dist(len(V)))


def mock_character_proposal(V, guide_spec, uniform=False):
    llm = _make_mock_llm(V, uniform)
    guide = _make_guide(guide_spec)

    return CharacterProposal(llm=llm, guide=guide)


def mock_token_proposal(V, guide_spec, K, uniform=False):
    llm = _make_mock_llm(V, uniform)
    guide = _make_guide(guide_spec)

    return TokenProposal(llm=llm, guide=guide, K=K)


def enumerate_traces(proposal, prompt, context):
    """
    This function uses program tracing and sampling without replacement to compute

        E_{(x,w) ~ q'}[ δ(x, x') * w ] = E_{(x,S) ~ q}[ δ(x, x') * w(x,S) ]
                                       = Σ_{x,S} δ(x, x') * q(x,S) * w(x,S)

    for each x' in V.

    Its use is to check whether our proposal satisfies properties like proper weighting through exact enumeration.
    """
    tracer = TraceSWOR()
    P = Float.chart()
    # sample without replacement until all traces have been exhausted
    while tracer.root.mass > 0:
        with tracer:
            (s, q, w) = asyncio.run(
                proposal.sample_next_token(draw=tracer, prompt=prompt, context=context)
            )
            P[s] += w * q
    return (P, tracer)


def enumerate_target(proposal, prompt, context):
    """
    This function exactly computes the unnormalized local POE target over next tokens given a prompt and context.
    """
    p_next = Float.chart()
    for token in proposal.llm.V:
        cfg_prob = 1
        for i, c in enumerate(token):
            cfg_prob *= proposal.guide.p_next(context + token[:i])[c]
        p_next[token] = cfg_prob * proposal.llm.p_next(prompt + context)[token]
    return p_next


def assert_proper_weighting(prompt, context, proposal, tol=1e-8):
    pi_q, _ = enumerate_traces(proposal, prompt, context)
    pi_true = enumerate_target(proposal, prompt, context)

    for x in proposal.llm.V:
        have = pi_q[x]
        want = pi_true[x]
        assert_equal(have, want, tol=tol)


def assert_unbiased_Z(prompt, context, proposal, tol=1e-8):
    pi_q, _ = enumerate_traces(proposal, prompt, context)
    pi_true = enumerate_target(proposal, prompt, context)

    have = pi_q.sum()
    want = pi_true.sum()
    assert_equal(have, want, tol=tol)
