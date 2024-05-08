from genparse.util import LarkStuff, hf_tokenizer
from genparse import CFGLM, locally_normalize
from genparse.align import CharAlignedCFGLM
from arsenal import timeit, colors
from time import time


def test_basic_aligned_model_arithmetic():

    lark_stuff = LarkStuff(r"""
    start: WS sum "</s>" | NAME "=" sum "</s>"

    sum: product | sum "+" product | sum MINUS product

    product: atom
        | product "*" atom
        | product "/" atom

    atom: NUMBER
            | MINUS atom
            | NAME
            | "(" sum ")"

    MINUS: /[\-]/
    NUMBER: /[\-+]?\d{1,3}(\.\d{1,3})?/
    WS: /[ \t\f\r\n]/
    NAME: /[a-zA-Z_]{1,5}/
    """)

    with timeit('grammar setup'):
        foo = lark_stuff.char_cfg(.9)
        foo = locally_normalize(foo, tol=1e-100).trim()
        assert len(foo) > 0
    with timeit('cfglm setup'):
        # the base character-level CFG language model
        lm = CFGLM(foo)

    with timeit('tokenizer setup'):
        H = hf_tokenizer()

    with timeit('CharAlignedCFGLM setup'):
        bpe_lm = CharAlignedCFGLM(lm=lm, words={x for _, x in H.pairs}, eos=H.tokenizer.eos_token)


    #===========================================================================
    # [2024-04-29 Mon] OPTIMIZATIONS: Tianyu provided the follow contexts, which
    # include a mix of slow and not so slow strings
    #
    #===========================================================================
    contexts = [
        #'Bey',
        #'BeyAk',
        'BeyAk=-',
        #'BeyAk=-amide',
        #'BeyAk=-amide*',
        #'BeyAk=-amide*pread',
        #'BeyAk=-amide*pread+(',
        #'BeyAk=-amide*pread+(Dust',
        #'BeyAk=-amide*pread+(Dusts',
        #'BeyAk=-amide*pread+(Dusts)--',
    ]

    for context in contexts:
        print(colors.yellow % repr(context))
        with timeit('parsing'):
            b4 = time()
            p = bpe_lm.p_next(context)
            took = time() - b4
        print('output size:', len(p))
        print('time/output:', took / len(p))


def test_basic_aligned_model_iql_small():

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

    with timeit('lark conversion'):
        foo = locally_normalize(lark_stuff.char_cfg(.1), tol=1e-100).trim()

    with timeit('tokenizer setup'):
        H = hf_tokenizer()

    with timeit('LM setup'):
        # the base character-level CFG language model
        lm = CFGLM(foo)
        bpe_lm = CharAlignedCFGLM(lm=lm, words={x for _, x in H.pairs},
                                  eos=H.tokenizer.eos_token)

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
