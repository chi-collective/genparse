from arsenal import colors
from arsenal.iterextras import take
from genparse import lark_guide, Float
from genparse.util import load_model_by_name
from genparse.proposal.crunch import Crunching
from time import time


def test_basic():
    llm = load_model_by_name('gpt2')
    guide = lark_guide(
        """
        start: "Sequential Monte Carlo is " ( "good" | "bad" | "awful" | "great" ) "!"
        """
    )

    q = Crunching(llm=llm, guide=guide)

    items = []
    start = time()
    for item in take(2, q.posterior_enumerate((llm.eos,))):
        print()
        print(item.ps, (colors.red % '·').join(item.ys))
        items.append(item)
    print('took', time() - start, 'sec')

    # check that items were returned in descending sorted order by probability
    want = sorted(items, key=lambda x: -x.ps)
    assert items == want, [items, want]

    # all sequences are valid
    assert all(guide(''.join(item.xs)) == 1 for item in items)

    # show posterior over tokenized strings
    P = Float.chart()
    for item in items:
        P[item.xs] += item.ps
    print(P.normalize().sort())

    # show posterior over strings
    P = Float.chart()
    for item in items:
        P[''.join(item.xs)] += item.ps
    P = P.normalize()
    print(P.sort())

    # This is a regression test; not an required behavior
    want = Float.chart(
        {
            'Sequential Monte Carlo is great!▪': 0.8498887881808378,
            'Sequential Monte Carlo is good!▪': 0.15011121181916223,
        }
    )
    want.assert_equal(P, tol=1e-5)


def test_basic_beam():
    llm = load_model_by_name('gpt2')
    guide = lark_guide(
        """
        start: "Sequential Monte Carlo is " ( "good" | "bad" | "awful" | "great" ) "!"
        """
    )

    q = Crunching(llm=llm, guide=guide)

    items = []
    start = time()
    for item in take(
        5, q.posterior_enumerate((llm.eos,), beam_width=1, max_generations=1)
    ):
        print()
        print(item.ps, (colors.red % '·').join(item.ys))
        items.append(item)
    print('took', time() - start, 'sec')

    # check that items were returned in descending sorted order by probability
    want = sorted(items, key=lambda x: -x.ps)
    assert items == want, [items, want]

    # all sequences are valid
    assert all(guide(''.join(item.xs)) == 1 for item in items)

    # show posterior over tokenized strings
    P = Float.chart()
    for item in items:
        P[item.xs] += item.ps
    print(P.normalize().sort())

    # show posterior over strings
    P = Float.chart()
    for item in items:
        P[''.join(item.xs)] += item.ps
    P = P.normalize()
    print(P.sort())

    # This is a regression test; not an required behavior
    want = Float.chart({'Sequential Monte Carlo is awful!▪': 1.0})
    want.assert_equal(P, tol=1e-5)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
