from genparse.util import LarkStuff, hf_tokenizer
from genparse import CFGLM, add_EOS, locally_normalize
from genparse.cfglm import CharAlignedCFGLM
from arsenal import timeit

import time
import numpy as np

from lark import Lark

def test_basic_aligned_model():

    grammar = r"""
    start: "SELECT" WS select_expr WS "FROM" WS from_expr [WS "WHERE" WS bool_condition] [WS "GROUP BY" WS var_list] [WS "ORDER BY" WS orderby_expr] WS EOS
    EOS: "</s>"
    select_expr: STAR | select_list
    bool_condition: bool_expr | "(" bool_condition WS "AND" WS bool_condition ")" | "(" bool_condition WS "OR" WS bool_condition ")"
    bool_expr: var "=" value | var ">" value | var "<" value
    from_expr: "data"
    orderby_expr: var_list WS "ASC" | var_list WS "DESC"
    select_list: select_var ("," WS select_var)*
    var_list: var ("," WS var)*
    select_var: var | "AVG(" var ")" | "MEDIAN(" var ")" | "COUNT(" var ")"
    var: "age" | "gender" | "year" | "state_color" | "zipcode" | "vote" | "race_ethnicity"
    value: NUMBER | "red" | "blue" | "white" | "black" | "latino" | "republican" | "democrat" | "male" | "female"
    STAR: "*"
    NUMBER: /\d+/
    //WS: /[ \t\f\r\n]/
    WS: " "
    """

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
        next_token_id = np.random.choice(a=len(p_list), p=np.array([x[1] for x in p_list]))
        next_token = p_list[next_token_id][0]

        production = production + next_token # max(p, key=p.get)
        print(production)
        if production.endswith("</s>"):
            break
        t_1 = time.time()
        step_time.append(t_1 - t_0)
    
    mean_step_time = np.mean(step_time)
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
