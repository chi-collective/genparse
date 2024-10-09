import numpy as np
from genparse import EOS
from genparse.util import InferenceSetup
from arsenal.maths import logsumexp, assert_equal


def test_boolean():
    grammar = """
        start: "a" "a" "a" | "b" "b" "b"
    """

    model = InferenceSetup('mock-gpt2', grammar, seed=0, num_processes=1)

    two_as = lambda p: ''.join(p.context)[:2] == 'aa'
    not_two_as_potential = lambda particles: [
        -np.inf if two_as(p) else 0 for p in particles
    ]

    approx = model(' ', method='smc', n_particles=100, potential=not_two_as_potential)

    print(approx)

    assert all(not two_as(p) or p.log_weight == -np.inf for p in approx), approx.particles

    approx = model(' ', method='is', n_particles=100, potential=not_two_as_potential)

    print(approx)

    assert all(not two_as(p) or p.log_weight == -np.inf for p in approx), approx.particles


def test_continous():
    grammar = 'start: "a"+'

    model = InferenceSetup(
        'mock-gpt2',
        grammar,
        seed=0,
        num_processes=1,
        proposal_name='token',
        proposal_opts={'K': None},
    )

    log_v = np.log(0.5)
    num_as = lambda context: ''.join(context).count('a')
    num_as_potential = lambda particles: [log_v * num_as(p.context) for p in particles]

    approx = model(
        prompt=' ',
        method='smc',
        n_particles=5,
        potential=num_as_potential,
        # do not resample but twist particles at each step
        # (twists not computed with ess_tresh = 0)
        ess_threshold=1e-100,
        max_tokens=4,
        return_record=True,
    )

    log_Z_1 = np.log(
        sum(
            # we coerce the eos token so we make sure to convert it here too
            p
            * model.guide.p_next_seq(
                context=(), extension=tuple(x) if x != model.llm.eos else (EOS,)
            )
            for x, p in model.llm.p_next(()).items()
        )
    )
    # Z is constant across contexts with length > 1 since we are using a uniform LM and guide
    log_Z_gt_1 = logsumexp([log_Z_1, model.llm.logp_next(('a',))[model.llm.eos]])

    for step in approx.record['history']:
        for particle in step['particles']:
            # weight (w/o twist) should be product of Zs since we are using token proposal with K = None
            log_w_t = log_Z_1 + (log_Z_gt_1 * (len(particle['context']) - 1))
            potential_t = log_v * num_as(particle['context'])

            have = particle['weight']
            want = log_w_t + potential_t

            assert_equal(have, want)

    approx = model(
        prompt=' ',
        method='smc',
        n_particles=5,
        potential=num_as_potential,
        ess_threshold=0.8,
        max_tokens=5,
        return_record=True,
    )

    r = -1
    avg_log_w_r = 0
    steps = approx.record['history']
    log_weight_updates = [log_Z_1] + ([log_Z_gt_1] * (len(steps) - 1))
    for i, step in enumerate(steps):
        for particle in step['particles']:
            context_len = len(particle['context'])
            log_w_r_1_t = sum(log_weight_updates[r + 1 : context_len])
            potential_t = log_v * num_as(particle['context'])
            potential_r_1 = log_v * num_as(particle['context'][: r + 1])

            have = particle['weight']
            want = avg_log_w_r + log_w_r_1_t + potential_t - potential_r_1

            assert_equal(want, have)

        if 'resample_indices' in step:
            r = i
            avg_log_w_r = step['average_weight']


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
