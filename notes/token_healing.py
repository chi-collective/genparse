"""
Experimental method for sampling a tokenization that is consistent with an observed string.
"""

from genparse import Float
from genparse.trace import TraceSWOR
from genparse.util import set_seed, load_model_by_name

from arsenal import colors
from arsenal.maths import sample_dict


def test_healing():
    set_seed(0)

    # lm = load_model_by_name('mock-gpt2')
    lm = load_model_by_name('gpt2', temperature=0.12)

    prompt = ' Sequential Monte Carlo is good'

    tracer = TraceSWOR()

    while tracer.root.mass > 1e-5:
        with tracer:
            context = sample_prompt(lm, prompt, verbosity=0, draw=tracer, complete=True)
        print(
            f'{tracer.root.mass:.8f} '
            + colors.light.cyan % '['
            + (colors.light.cyan % '|').join(context)
            + colors.light.cyan % ']'
        )

    print(f'truncated with {tracer.root.mass:g} mass left')


# TODO: This is a work-in-progress method for sampling a tokenization that is
# consistent with an observed string `prompt`
def sample_prompt(self, prompt, draw=sample_dict, verbosity=0, complete=False):
    # context = (self.tokenizer.bos_token,)
    context = ()
    k = 0
    while True:
        p = self.p_next(context)

        if verbosity > 0:
            print(context, repr(prompt[k:]))
        q = Float.chart()
        for y in self.V:
            if prompt[k:].startswith(y):
                q[y] = p[y]

            if not complete and len(prompt[k:]) <= len(y) and y.startswith(prompt[k:]):
                q[y] = p[y]

        # EOS is not allowed until after the prompt is covered
        if len(prompt[k:]) == 0:
            q[self.eos] = p[self.eos]

            if complete:
                q.clear()
                q[self.eos] = 1

        q = q.normalize()

        if verbosity > 1:
            print(q.top(5))

        q = q.sort_descending()
        y = draw(q)

        context = context + (y,)

        if y == self.eos:
            return context

        k += len(y)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
