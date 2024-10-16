from genparse import EOS
from genparse.util import InferenceSetup

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def _get_step_num(context, context_step_condition):
    return sum(context_step_condition(context[:i]) for i in range(1, len(context) + 1))


def _check_incremental_step_count(record, context_step_condition):
    for step in record['history']:
        want = step['step']
        for p in step['particles']:
            if p['context'][-1] == EOS:
                continue
            have = _get_step_num(p['context'], context_step_condition)
            assert have == want, [p, have, want]


def test_basic():
    context_step_condition = lambda context: context[-1] == '\n'
    step_condition = (
        lambda particle: context_step_condition(particle.context) or particle.done
    )

    grammar = """
    start: line line+
    line: "a"+"\\n"
    """
    model = InferenceSetup('mock-gpt2', grammar=grammar, seed=0)
    approx = model(
        ' ',
        n_particles=10,
        verbosity=1,
        step_condition=step_condition,
        return_record=True,
    )
    model.cleanup()

    assert all((not p.done) or step_condition(p) for p in approx.particles)

    _check_incremental_step_count(approx.record, context_step_condition)


def test_basic_vllm():
    context_step_condition = lambda context: context[-1] == '\n'
    step_condition = (
        lambda particle: context_step_condition(particle.context) or particle.done
    )

    grammar = """
    start: line line+
    line: "a"+"\\n"
    """
    model = InferenceSetup('gpt2', grammar=grammar, seed=0)
    approx = model(
        ' ',
        n_particles=10,
        verbosity=1,
        step_condition=step_condition,
        return_record=True,
    )
    model.cleanup()

    assert all((not p.done) or step_condition(p) for p in approx.particles)

    _check_incremental_step_count(approx.record, context_step_condition)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
