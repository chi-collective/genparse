import pandas as pd
import numpy as np
from collections import Counter
from arsenal import colors
from arsenal.maths import compare

from genparse.steer import LocalProduct, run, normalize
from genparse.align import pullback
from genparse.cfglm import Float, CFG, CFGLM, add_EOS, explode
from genparse.lm import AutoTokenizer, AutoModelForCausalLM, NoCacheGPT, TokenGPT2


# TODO: this doesn't have an actual test, so it is just a test of the interface
def test_pullback_gpt():

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")
    llm = NoCacheGPT(gpt_model)

#    llm = TokenGPT2(gpt_model)

    out = pullback(llm, ' th', tokenizer.decode, tuple(tokenizer.encode(' ')))
    print('>>>', out)

    print(pd.DataFrame(out.items()).sort_values(1, ascending=False).head(20))


def test_pullback_cfg():

    cfg = CFG.from_string("""

    1: S -> A A A A
    1: A -> a
    1: A -> aa
    1: A -> b
    1: A -> bb

    """, Float)

    char_lm = CFGLM(explode(cfg).cnf)

    prefix = 'aa'

    want = char_lm.p_next(prefix)

    have = pullback(char_lm, prefix)

    print('want=', want)
    print('have=', have)

    assert have.metric(want) <= 1e-5


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
