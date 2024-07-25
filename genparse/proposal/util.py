import numpy as np
import asyncio
from arsenal.maths import random_dist, assert_equal

from genparse.trace import TraceSWOR
from genparse import Float
from genparse.lm import MockLLM, LM
from genparse.proposal import TokenProposal, CharacterProposal
from genparse.util import lark_guide


def _make_guide(guide_spec):
    if isinstance(guide_spec, str):
        return lark_guide(guide_spec)
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


def enumerate_target(proposal, prompt, context):
    """
    Computes the unnormalized local product of experts target over next tokens
    given a `prompt` and `context`.
    """
    p_next = Float.chart()
    p_next_llm = proposal.llm.p_next(prompt + context)
    for token in proposal.llm.V:
        p_next[token] = p_next_llm[token] * proposal.guide.p_next_seq(
            ''.join(context), token
        )
    return p_next


def assert_proper_weighting(prompt, context, proposal, tol=1e-8):
    pi_q, _ = proposal.enumerate_traces(prompt, context)
    pi_true = enumerate_target(proposal, prompt, context)

    for x in proposal.llm.V:
        have = pi_q[x]
        want = pi_true[x]
        assert_equal(have, want, tol=tol)


def assert_unbiased_Z(prompt, context, proposal, tol=1e-8):
    pi_q, _ = proposal.enumerate_traces(prompt, context)
    pi_true = enumerate_target(proposal, prompt, context)

    have = pi_q.sum()
    want = pi_true.sum()
    assert_equal(have, want, tol=tol)
