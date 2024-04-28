from genparse.util import LarkStuff, hf_tokenizer
from genparse import CFGLM, add_EOS, locally_normalize
from genparse.cfglm import CharAlignedCFGLM
from arsenal import timeit


def test_basic_aligned_model():

    lark_stuff = LarkStuff(r"""
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
    """)

    foo = lark_stuff.char_cfg(.1)
    foo = locally_normalize(foo, tol=1e-100).trim()
    assert len(foo) > 0

    H = hf_tokenizer()

    # the base character-level CFG language model
    lm = CFGLM(add_EOS(foo))

    bpe_lm = CharAlignedCFGLM(lm=lm, words={x for _, x in H.pairs}, eos=H.tokenizer.eos_token)

    with timeit('took'):

        p = bpe_lm.p_next('')
        print(p)
        assert p.keys() == {'S', 'SE', 'SELECT'}

        p = bpe_lm.p_next('SELECT * FROM data')
        print(p)
        assert p.keys() == {' ', ' <', ' </', ' G', ' W', ' O', ' GR', ' WH', ' OR',
                            ' GROUP', ' WHERE', ' ORDER'}

        p = bpe_lm.p_next('SELECT age FROM data')
        print(p)
        assert p.keys() == {' ', ' <', ' </', ' G', ' W', ' O', ' GR', ' WH', ' OR',
                            ' GROUP', ' WHERE', ' ORDER'}

#    # Note: We generate the underlying *token* sequences without replacement,
#    # but the *character* sequences may be repeated (in limited ways).
#    import random, numpy as np
#    random.seed(0); np.random.seed(0)
#    from genparse.inference import TraceSWOR
#    tracer = TraceSWOR()
#    distinct = set()
#    with timeit('took'):
#        for t in range(100):
#            if t % 100 == 0: print(t, tracer.root.mass)
#            with tracer:
#                ys = bpe_lm.sample(draw=tracer)
#            if ys not in distinct:
#                print(ys)
#            distinct.add(ys)

#===============================================================================
# [2024-04-28 Sun] Optimizations:
#
# initial version (best of several runs)  16.7135 sec
#
# using integers for nonterminals:        14.4915 sec;  1.15x faster
#
# using lists of charts to avoid copying: 13.0670 sec;  1.1x faster
#
# left-child loop in `extend_chart`)       6.0260 sec;  2.2x faster
# left-child loop in `next_token_weights`  1.6176 sec;  3.7x faster
#===============================================================================

if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
