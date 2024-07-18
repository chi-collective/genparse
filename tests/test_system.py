from genparse import InferenceSetup
from arsenal import colors


def test_basic():
    grammar = """
    start: "Sequential Monte Carlo is " ( "good" | "bad" )

    %ignore /[ ]/
    """
    infer = InferenceSetup('gpt2', grammar, proposal_name='character')
    particles = infer('', n_particles=15, return_record=True, seed=1234)

    print(particles)

    def cost(x, y):
        X = set(x)
        Y = set(y)
        return len(X & Y) / len(X | Y)

    candidate = 'Seq MC is good'
    print('candidate:', repr(candidate), 'risk:', particles.risk(cost, candidate))

    assert particles.record is not None
    f = '/tmp/viz.html'
    particles.record.plotly().write_html(f)
    print(f'wrote {colors.link("file://" + f)}')


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
