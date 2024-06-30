from genparse.trace import TraceSWOR


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


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
