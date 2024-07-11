import pytest
import numpy as np
from genparse.lm import LM
from genparse.util import load_model_by_name
from arsenal import colors


def test_tokenized_llm():
    lm = load_model_by_name('gpt2')

    with pytest.raises(AssertionError, match='`context` must be explicitly tokenized'):
        lm('Sequential Monte Carlo is good')

    tokens = lm.encode_prompt('Sequential Monte Carlo is')
    assert tokens == ('Sequ', 'ential', ' Monte', ' Carlo', ' is')

    # demonstrate the tokenization inconsistency at the end of the prompt;
    # at least for this tokenizer, all but the last token are stable
    tokens = lm.encode_prompt('Sequential Monte Carlo is ')
    assert tokens == ('Sequ', 'ential', ' Monte', ' Carlo', ' is', ' ')

    tokens = lm.encode_prompt('Sequential Monte Carlo is good')
    assert tokens == ('Sequ', 'ential', ' Monte', ' Carlo', ' is', ' good')

    # regression test: gpt2 has made up its mind about Sequential Monte Carlo
    good = lm.encode_prompt('Sequential Monte Carlo is good') + (lm.tokenizer.eos_token,)
    bad = lm.encode_prompt('Sequential Monte Carlo is bad') + (lm.tokenizer.eos_token,)
    assert lm(good) > lm(bad), [lm(good), lm(bad)]

    # missing EOS
    with pytest.raises(AssertionError, match='Context must end with eos .*'):
        assert lm(lm.encode_prompt('Sequential Monte Carlo is bad'))

    with pytest.raises(AssertionError, match='OOVs detected.*'):
        assert lm(('Sequential', 'Monte Carlo is bad'))
    with pytest.raises(AssertionError, match='OOVs detected.*'):
        assert lm.p_next(('Sequential', 'Monte Carlo is bad'))

    # regression test, not necessarily a desired behavior
    have = lm.p_next(('Sequ', 'ential', ' Monte', ' Carlo', ' is')).materialize(top=10)
    want = {
        ' a': 0.632543329550616,
        ' an': 0.15710208066783773,
        ' the': 0.1204166775353927,
        ' one': 0.03018382983875458,
        ' not': 0.015445944924912892,
        ' used': 0.014886780312176533,
        ' based': 0.009135932027301829,
        ' another': 0.007827981830716771,
        ' now': 0.006395742653323747,
        ' also': 0.006061700658967281,
    }
    have.assert_equal(want, tol=1e-5)


def test_llm_probs():
    lm = load_model_by_name('gpt2')

    x = lm.encode_prompt('Sequential Monte Carlo is good') + (lm.tokenizer.eos_token,)

    print('scoring', x)

    have = np.log(LM.__call__(lm, x))
    want = np.log(lm(x))

    print(
        'prob:',
        have,
        want,
    )

    # check consistency between the two methods for evaluating the LM's probability
    assert np.allclose(have, want)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
