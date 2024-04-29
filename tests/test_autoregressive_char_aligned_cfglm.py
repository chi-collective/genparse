from genparse.util import LarkStuff, hf_tokenizer
from genparse import CFGLM, add_EOS, locally_normalize
from genparse.cfglm import CharAlignedCFGLM
from arsenal import timeit

import time
import numpy as np

from lark import Lark

def test_basic_aligned_model():

    grammar_id = "basic_calc" # "basic_calc" # "restricted_sql" "simple_json"
    grammar_dir = f"grammars/{grammar_id}.lark"

    grammar = open(grammar_dir).read()

    lark_stuff = LarkStuff(grammar)
    lark_parser = Lark(grammar)

    foo = lark_stuff.char_cfg(.1)
    foo = locally_normalize(foo, tol=1e-100).trim()
    assert len(foo) > 0

    H = hf_tokenizer()

    # the base character-level CFG language model
    lm = CFGLM(add_EOS(foo))

    bpe_lm = CharAlignedCFGLM(lm=lm, words={x for _, x in H.pairs}, eos=H.tokenizer.eos_token)

    np.random.seed(123)
    production = ""
    step_time = []

    while True:
        t_0 = time.time()
        p = bpe_lm.p_next(production)
        p_list = list(p.items())
        # next_token_id = np.random.choice(a=len(p_list), p=np.array([x[1] for x in p_list]))
        next_token_id = np.random.choice(a=len(p_list), p=np.full(len(p_list), 1/len(p_list)))
        next_token = p_list[next_token_id][0]

        production = production + next_token # max(p, key=p.get)
        if production.endswith("</s>"):
            break
        t_1 = time.time()
        print(production, t_1 - t_0)
        step_time.append(t_1 - t_0)
    
    mean_step_time = np.mean(step_time[1:])
    print(step_time)
    print(f"Mean step time: {mean_step_time}")

    success, failure = 0, 0
    try:
        parse_tree = lark_parser.parse(production)
        success += 1
        # print(parse_tree)
    except Exception as e:
        print(e)
        print(production, "failed")
        failure += 1
    
    print(f"Success: {success}, Failure: {failure}")

if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
