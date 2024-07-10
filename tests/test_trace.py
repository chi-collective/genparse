import pytest
from tqdm import tqdm
from arsenal.maths import sample

import os
import psutil
from contextlib import contextmanager

from genparse import EarleyLM as CFGLM, MockLLM, EOS
from genparse.lm import LazyProb
from genparse.trace import Node, TraceSWOR


@contextmanager
def memory_change():
    pid = os.getpid()
    before = psutil.Process(pid).memory_info().rss
    yield
    after = psutil.Process(pid).memory_info().rss
    print(f'Used mem: {(after - before) / 10**6:.2f} MB')


def test_basics():
    p = {'red': 1 / 3, 'green': 1 / 3, 'blue': 1 / 3}

    trace = TraceSWOR()

    sampled = set()
    while trace.root.mass > 0:
        with trace:
            sampled.add(trace(p))
    assert sampled == {'red', 'green', 'blue'}

    # called just for coverage, no assertions made.
    trace.root.graphviz()
    trace.sixel_render()


def test_grammar():
    cfg = CFGLM.from_string(
        """
        1: S -> a
        1: S -> a a
        2: S -> a a a
        """
    )
    Node.global_node_counter = 0
    tracer = TraceSWOR()
    while tracer.root.mass > 0:
        with tracer:
            cfg.sample(draw=tracer)
    tracer.sixel_render()
    assert sum(1 for _ in tracer.root.downstream_nodes()) == 7
    assert (
        sum(1 for x in tracer.root.downstream_nodes() if x.child_masses is not None) == 4
    )


def test_mock_lm():
    lm = MockLLM([EOS, 'a', 'b', 'c'], EOS)
    tracer = TraceSWOR()
    while tracer.root.mass > 0.0:
        with tracer:
            lm.sample(draw=tracer, max_tokens=1)
    tracer.sixel_render()
    assert len(list(tracer.root.downstream_nodes())) == 17
    assert (
        sum(1 for x in tracer.root.downstream_nodes() if x.child_masses is not None) == 4
    )


def sample_lazyprob(p):
    if not isinstance(p, LazyProb):
        raise ValueError(f'Expected LazyProb, got {type(p)}')
    index = sample(p.values())
    tok = p.keys()[index]
    return tok


@pytest.mark.skip  # performance test
def test_deep_no_tree():
    lm = MockLLM({EOS}.union(str(n) for n in range(50000)), EOS)
    tracer = TraceSWOR()
    with memory_change():
        for leaves in (progress := tqdm(range(1, 201))):
            progress.set_description(
                f'{leaves} generations for {tracer.root.mass:.15e} remaining mass'
            )
            lm.sample(draw=sample_lazyprob, max_tokens=50)


@pytest.mark.skip  # performance test
def test_deep():
    lm = MockLLM({EOS}.union(str(n) for n in range(50000)), EOS)
    tracer = TraceSWOR()
    with memory_change():
        for leaves in (progress := tqdm(range(1, 201))):
            progress.set_description(
                f'{leaves} generations for {tracer.root.mass:.15e} remaining mass'
            )
            with tracer:
                lm.sample(draw=tracer, max_tokens=50)
            leaves += 1


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
