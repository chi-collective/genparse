"""
Exploring Grammar Aligned Decoding (Park et al., 2024).  We extend GAD to
support probabilistic grammars.
"""

import numpy as np
from arsenal import colors
from arsenal.maths import sample_dict

from genparse import Float


class DeadEnd(Exception):
    pass


class AdaptiveSampler:
    def __init__(self, lm1, guide):
        self.lm1 = lm1
        self.guide = guide
        self.root = Node(mass=1.0, parent=None, context='')
        self.curr = self.root

    def update(self, curr, verbosity=0):
        if curr is None:
            return
        elif curr.children is None:  # base case
            pass
        else:
            # recursive rule: update the sum-of-child-mass invariant
            new_mass = sum(curr.p1[a] * y.mass for a, y in curr.children.items())
            assert (
                curr.mass >= new_mass
            ), 'Not an overestimate; failed {curr.mass=} >= {new_mass=}'
            curr.mass = new_mass
        self.update(curr.parent)

    def __call__(self, draw=sample_dict):
        "Sample an action while updating the trace cursor and tree data structure."

        curr = self.curr

        if curr.children is None:  # initialize the newly discovered node
            # print(colors.light.cyan % 'new node', curr.context)
            p1 = self.lm1.p_next(curr.context)
            p2 = self.guide.p_next(curr.context)
            # p2 = self.guide.model.next_token_weights(self.guide.model.chart(curr.context))
            curr.p1 = p1
            curr.p2 = p2

            curr.children = {
                a: Node(p2[a], parent=curr, context=curr.context + a)
                for a in p1
                if p1[a] * p2[a] > 0
            }

        # Note: the sampling policy is completely arbitrary; it is inefficient
        # to sample on-policy like we are doing here because we will continue
        # sampling paths in the tree even after their subtrees have converged.
        if not curr.children:
            raise DeadEnd()
        p = Float.chart({a: curr.p1[a] * y.mass for a, y in curr.children.items()})

        a = draw(p)

        self.curr = curr.children[a]  # advance the cursor
        return a

    def compare(self, x, oracle, P, verbosity):
        "Traverse the tree; check if the conditional probability matches the oracle"

        tol = 1e-10

        if x.children is None:
            assert abs(self.guide(x.context) - x.mass) <= tol
            return

        if verbosity > 0:
            # TODO: We sill don't know the precise relationship between `x.mass`
            # and values obtained from the oracle grammar.
            #
            # This candiate seems plausible and had passed several tests.
            candidate = oracle.cfg.prefix_grammar(
                x.context
            ) / self.lm1.cfg.prefix_grammar(x.context)

            print(
                '  ' * len(x.context),
                colors.orange % 'compare:',
                repr(x.context),
                x.mass,
                candidate,
                colors.mark(abs(x.mass - candidate) <= tol),
            )

        want = oracle.p_next(x.context)

        have = Float.chart()
        if x.mass > 0:
            for a, y in x.children.items():
                have[a] = x.p1[a] * y.mass / x.mass

        # check the invariant
        new_mass = sum(x.p1[a] * y.mass for a, y in x.children.items())
        assert abs(x.mass - new_mass) <= tol

        err = have.metric(want)

        # if verbosity > 0:
        #    print(colors.mark(err <= tol), f'{err=} {have=}')
        #    if err > tol: print(f'{want=}')
        #    #print(f'Z: {x.mass=}, {Z=}')
        assert err <= tol, f'{err=} {have=} {want=}'

        for a, y in x.children.items():
            self.compare(y, oracle=oracle, P=have[a] * P, verbosity=verbosity)


class Node:
    def __init__(self, mass, context, parent):
        self.mass = mass
        self.context = context
        self.parent = parent
        self.p1 = None
        self.p2 = None
        self.children = None

    def show(self, indent=''):
        if self.children is None:
            return
        for a, x in self.children.items():
            print(f'{indent}{a} ({x.mass/self.mass if self.mass > 0 else 0})')
            x.show(indent + '  ')


def test_basics():
    from genparse import WFSA, EarleyLM, EOS

    np.random.seed(0)

    lm1 = EarleyLM.from_string(
        """
        0.25: S -> a S a
        0.25: S -> b S b
        0.5:  S ->
        """
    )

    wfsa = (
        (
            WFSA.from_string('aaaa', Float, w=0.5)
            + WFSA.from_string('abb', Float, w=0.2)
            + WFSA.from_string('bbbbbb', Float, w=0.3)
        )
        * WFSA.from_string((EOS,), Float, w=1)
    ).renumber

    wfsa_cfg = wfsa.to_cfg().trim().renumber()

    guide = EarleyLM(wfsa_cfg)

    oracle = EarleyLM((lm1.cfg @ wfsa).trim())
    print('total=', (lm1.cfg @ wfsa).trim().treesum())
    C = oracle.cfg.cnf.language(12).sum()
    print('C=', C)
    print(oracle.cfg.cnf.language(12).normalize())

    sampler = AdaptiveSampler(lm1=lm1, guide=guide)

    for iteration in range(1, 50):
        print(colors.light.yellow % '\nIteration', iteration)

        sampler.curr = sampler.root
        while True:
            try:
                x = sampler()
            except DeadEnd:
                break  # sampler hit a dead end
            print(sampler.curr.context)
            if x == EOS:
                break

        if sampler.curr.context[-1] == EOS:
            sampler.curr.mass = sampler.guide(
                sampler.curr.context
            )  # whole string probability
        else:
            sampler.curr.mass = 0  # dead ends have probability zero

        sampler.update(sampler.curr)
        sampler.curr = sampler.root

    # sampler.root.show()

    sampler.compare(x=sampler.root, P=1, oracle=oracle, verbosity=1)

    print(oracle.cfg.cnf.language(12).normalize())


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
