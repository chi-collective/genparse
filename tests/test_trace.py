from genparse.lm import LazyProb
from genparse.trace import Node, TraceSWOR
from tqdm import tqdm
import numpy as np
from arsenal.maths import sample
import pytest


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
    from genparse import EarleyLM as CFGLM

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
            s, p = cfg.sample(draw=tracer)
    tracer.sixel_render()
    assert Node.global_node_counter == 7
    assert tracer.inner_nodes == 4


def test_mock_lm():
    from genparse import MockLLM, EOS

    np.random.seed(0)

    lm = MockLLM([EOS, 'a', 'b', 'c'], EOS)
    Node.global_node_counter = 0
    tracer = TraceSWOR()
    while tracer.root.mass > 0.0:
        with tracer:
            s, p = lm.sample(draw=tracer, max_tokens=1)
    tracer.sixel_render()
    assert Node.global_node_counter == 17
    assert tracer.inner_nodes == 4


def sample_lazyprob(p):
    if not isinstance(p, LazyProb):
        raise ValueError(f'Expected LazyProb, got {type(p)}')
    index = sample(p.values())
    tok = p.keys()[index]
    return tok


@pytest.mark.skip  # performance test
def test_deep_no_tree():
    from genparse import MockLLM, EOS
    import os
    import psutil

    Node.global_node_counter = 0
    lm = MockLLM({EOS}.union(str(n) for n in range(50000)), EOS)
    tracer = TraceSWOR()
    for leaves in (progress := tqdm(range(1, 201))):
        progress.set_description(
            f'{leaves} generations for {tracer.root.mass:.15e} remaining mass, {Node.global_node_counter} nodes, {tracer.inner_nodes} inner nodes'
        )
        s, p = lm.sample(draw=sample_lazyprob, max_tokens=50)
    print(f'Used mem: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB')


@pytest.mark.skip  # performance test
def test_deep():
    from genparse import MockLLM, EOS
    import os
    import psutil

    lm = MockLLM({EOS}.union(str(n) for n in range(50000)), EOS)
    Node.global_node_counter = 0
    tracer = TraceSWOR()
    for leaves in (progress := tqdm(range(1, 201))):
        progress.set_description(
            f'{leaves} generations for {tracer.root.mass:.15e} remaining mass, {Node.global_node_counter} nodes, {tracer.inner_nodes} inner nodes'
        )
        with tracer:
            s, p = lm.sample(draw=tracer, max_tokens=50)
        leaves += 1
    print(f'Used mem: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB')


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
