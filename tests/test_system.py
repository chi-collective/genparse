import numpy as np

from arsenal import timeit, colors
import pickle

from genparse import Float
from genparse.util import InferenceSetup


def test_basic():
    grammar = """
    start: "Sequential Monte Carlo is " ( "good" | "bad" )
    """
    infer = InferenceSetup('gpt2', grammar, proposal_name='character')
    particles = infer(' ', n_particles=15)

    print(particles)
    # {
    #  'Sequential Monte Carlo is good▪': 0.7770842914205952,
    #  'Sequential Monte Carlo is bad▪': 0.22291570857940482,
    # }


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
