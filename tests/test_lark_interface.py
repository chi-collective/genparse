import numpy as np

from arsenal.maths import compare
from collections import Counter

from genparse.util import LarkStuff
from genparse import CFGLM, add_EOS, locally_normalize
from genparse.util import normalize


grammar1 = r"""
start: WS? "SELECT" WS select_expr WS "FROM" WS from_expr [WS "WHERE" WS bool_condition] [WS "GROUP BY" WS var_list] [WS "ORDER BY" WS orderby_expr] WS EOS
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
WS: /[ \t\f\r\n]/
"""


def test_tokenization_basics():

    lark_stuff = LarkStuff(grammar1)

    text = 'SELECT state_color FROM data </s>'
    tokens = list(lark_stuff.lex(text))

    T = lark_stuff.transducer(ignore='')   # `ignore` specific to this grammar
    tks = T(text, None).renumber.epsremove.trim
    #print(tks)

    # check that lark's token sequence is in the transducer's language
    tmp = tks.to_cfg().renumber().cnf.language(10)

    target = tuple(t.type for t in tokens)
    assert target in tmp


def test_parsing_basics():

    lark_stuff = LarkStuff(grammar1)

    text = 'SELECT state_color FROM data </s>'
    tokens = list(lark_stuff.lex(text))

    g = lark_stuff.convert().renumber()
    assert g.in_cnf()    # lark returns a grammar in CNF


    tokens = ['WS', 'SELECT', 'WS', 'ZIPCODE', 'WS', 'FROM', 'WS', 'DATA', 'WS', 'EOS']

    assert g(tokens) > 0

    #print(g.cnf.prefix_grammar.trim().cnf)
    tokens = ['WS', 'SELECT', 'WS', 'ZIPCODE', 'WS', 'FROM', 'WS', 'DATA']

    assert g.prefix_weight(tokens) > 0


def test_char_level_cfg():
    lark_stuff = LarkStuff(grammar1)

    # this grammar is kind of silly - it requires a space at the front of the string
    text = 'SELECT state_color FROM data </s>'

    #tokens = list(lark_stuff.lex(text))
    #print(lark_stuff.parser.parse(tokens, 'start'))

    cfg = lark_stuff.char_cfg(.1)

    #print(len(cfg.trim()))
    #print(len(cfg.cnf))

    assert cfg(text) > 0

    lm = CFGLM(add_EOS(locally_normalize(cfg, tol=1e-40, maxiter=np.inf)))

    p = normalize(lm.p_next('SELECT state_color FROM '))
    print(p)
    p.assert_equal({'d': 1})

    p = normalize(lm.p_next('S'))
    print(p)
    p.assert_equal({'E': 1})

    p = normalize(lm.p_next('SELECT '))
    print(p)
    assert p.argmax() == '*'


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
