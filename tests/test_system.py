from genparse.util import InferenceSetup


def test_basic():
    grammar = """
    start: "Sequential Monte Carlo is " ( "good" | "bad" )
    """
    infer = InferenceSetup('gpt2', grammar, proposal_name='character')
    particles = infer(' ', n_particles=15, return_record=True)

    print(particles)

    def cost(x, y):
        X = set(x)
        Y = set(y)
        return len(X & Y) / len(X | Y)

    candidate = 'Seq MC is good'
    print('candidate:', repr(candidate), 'risk:', particles.risk(cost, candidate))

    assert particles.record is not None


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
