import torch
import warnings
from genparse.util import InferenceSetup

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def test_lms():
    grammar = 'start: "a" "b" "c"'
    if torch.cuda.is_available():
        model = InferenceSetup('gpt2', grammar=grammar, seed=0)
        model(' ', n_particles=10, verbosity=1)
        model.cleanup()
    else:
        warnings.warn('Skipping vllm test because cuda is not available.')

    model = InferenceSetup('mock-gpt2', grammar=grammar, seed=0)
    model(' ', n_particles=10, verbosity=1)
    model.cleanup()


def test_proposals():
    grammar = 'start: "a" "b" "c"'

    model = InferenceSetup(
        'mock-gpt2', grammar=grammar, proposal_name='character', seed=0
    )
    model(' ', n_particles=10, verbosity=1)
    model.cleanup()

    model = InferenceSetup(
        'mock-gpt2',
        grammar=grammar,
        proposal_name='token',
        proposal_opts={'K': 10},
        seed=0,
    )
    model(' ', n_particles=10, verbosity=1)
    model.cleanup()

    model = InferenceSetup(
        'mock-gpt2', grammar=grammar, proposal_name='character', num_processes=1, seed=0
    )
    model(' ', n_particles=10, verbosity=1)
    model.cleanup()


def test_parsers():
    grammar = 'start: "a" "b" "c"'

    model = InferenceSetup('mock-gpt2', grammar=grammar, use_rust_parser=True, seed=0)
    model(' ', n_particles=10, verbosity=1)
    model.cleanup()

    model = InferenceSetup('mock-gpt2', grammar=grammar, use_rust_parser=False, seed=0)
    model(' ', n_particles=10, verbosity=1)
    model.cleanup()


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
