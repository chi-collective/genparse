import numpy as np

from genparse.parse.earley import EarleyLM, Earley
from genparse.cfglm import BoolCFGLM, locally_normalize
from genparse.util import LarkStuff, expand_case_insensitive


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

    T = lark_stuff.transducer(ignore='')  # `ignore` specific to this grammar
    tks = T(text, None).renumber.epsremove.trim
    # print(tks)

    # check that lark's token sequence is in the transducer's language
    tmp = tks.to_cfg().renumber().cnf.language(10)

    target = tuple(t.type for t in tokens)
    assert target in tmp


def test_parsing_basics():
    lark_stuff = LarkStuff(grammar1, cnf=True)

    text = 'SELECT state_color FROM data </s>'
    tokens = list(lark_stuff.lex(text))

    g = lark_stuff.convert().renumber()
    assert g.in_cnf()  # lark returns a grammar in CNF

    tokens = ['WS', 'SELECT', 'WS', 'ZIPCODE', 'WS', 'FROM', 'WS', 'DATA', 'WS', 'EOS']

    assert g(tokens) > 0

    # print(g.cnf.prefix_grammar.trim().cnf)
    tokens = ['WS', 'SELECT', 'WS', 'ZIPCODE', 'WS', 'FROM', 'WS', 'DATA']

    assert g.prefix_weight(tokens) > 0

    ####
    # Now, we repeat the same as above without CNF conversion
    #
    # NOTE: the grammars are unfortunately not equivalent because of how we
    # assigned them weights.

    lark_stuff = LarkStuff(grammar1, cnf=False)

    text = 'SELECT state_color FROM data </s>'
    tokens = list(lark_stuff.lex(text))

    g = lark_stuff.convert().renumber()

    tokens = ['WS', 'SELECT', 'WS', 'ZIPCODE', 'WS', 'FROM', 'WS', 'DATA', 'WS', 'EOS']

    assert Earley(g)(tokens) > 0

    # print(g.cnf.prefix_grammar.trim().cnf)
    tokens = ['WS', 'SELECT', 'WS', 'ZIPCODE', 'WS', 'FROM', 'WS', 'DATA']

    assert Earley(g.prefix_grammar)(tokens) > 0


def test_char_level_cfg():
    lark_stuff = LarkStuff(grammar1)

    # this grammar is kind of silly - it requires a space at the front of the string
    text = 'SELECT state_color FROM data </s>'

    # tokens = list(lark_stuff.lex(text))
    # print(lark_stuff.parser.parse(tokens, 'start'))

    cfg = lark_stuff.char_cfg(0.1)

    # print(len(cfg.trim()))
    # print(len(cfg.cnf))

    assert cfg(text) > 0

    lm = EarleyLM(locally_normalize(cfg, tol=1e-40, maxiter=np.inf))

    p = lm.p_next('SELECT state_color FROM ').normalize()
    print(p)
    p.assert_equal({'d': 1})

    p = lm.p_next('S').normalize()
    print(p)
    p.assert_equal({'E': 1})

    p = lm.p_next('SELECT ').normalize()
    print(p)
    assert p.argmax() == '*'


def test_char_lm_basics1():
    lark_stuff = LarkStuff(
        r"""

    start: "SELECT" WS NAME WS "FROM" WS NAME WS EOS

    EOS: "</s>"
    NAME: /[A-Za-z][A-Za-z]?[A-Za-z]?[A-Za-z]?[A-Za-z]?/
    STAR: "*"
    WS: /[ ]/

    """
    )

    cfg = lark_stuff.convert().renumber()
    c2t = lark_stuff.transducer(ignore='', decay=0.3)
    cfg_t = (c2t.renumber @ cfg).trim()

    # pg = cfg_t.cnf.trim().prefix_grammar.trim()

    pg = locally_normalize(cfg_t.cnf.trim()).prefix_grammar.trim()
    pg = pg.cnf

    assert pg('S') > 0
    assert pg('SEL') > 0


def test_char_lm_basics2():
    lark_stuff = LarkStuff(
        r"""

    start: NAME

    NAME: /(a|b)+/

    """
    )

    cfg = lark_stuff.convert().renumber()
    c2t = lark_stuff.transducer(ignore='', decay=0.1).renumber.trim
    cfg_t = (c2t @ cfg).trim()

    # print(cfg.cnf.language(5))
    # print(cfg_t.cnf.language(3))

    # print(cfg_t.agenda().__str__(style_value=lambda k, v: (colors.light.red % v) if v > 1 or v < 0 else v))

    pg = cfg_t.cnf.prefix_grammar.cnf.trim()

    pg.cnf.language(3).assert_equal(
        {
            (): 0.025,
            ('a',): 0.0125,
            ('b',): 0.0125,
            ('a', 'a'): 0.00125,
            ('a', 'b'): 0.00125,
            ('b', 'a'): 0.00125,
            ('b', 'b'): 0.00125,
            ('a', 'a', 'a'): 0.000125,
            ('a', 'a', 'b'): 0.000125,
            ('a', 'b', 'a'): 0.000125,
            ('a', 'b', 'b'): 0.000125,
            ('b', 'a', 'a'): 0.000125,
            ('b', 'a', 'b'): 0.000125,
            ('b', 'b', 'a'): 0.000125,
            ('b', 'b', 'b'): 0.000125,
        },
        tol=1e-4,
    )

    assert pg('a') > 0


def test_char_lm_basics3():
    lark_stuff = LarkStuff(
        r"""

    start: "SELECT" " " NAME " " "FROM"

    NAME: /b+/

    """
    )

    cfg = lark_stuff.convert().renumber()
    c2t = lark_stuff.transducer(ignore='', decay=0.1).renumber
    cfg_t = (c2t @ cfg).trim()

    cfg_t_lm = EarleyLM(locally_normalize(cfg_t, tol=1e-50))

    v = cfg_t_lm.p_next('SELECT bb').normalize()
    print(v)
    assert set(v.trim().keys()) == {' ', 'b'}

    del cfg, c2t, cfg_t

    char_cfg = lark_stuff.char_cfg(0.1)
    char_lm = EarleyLM(locally_normalize(char_cfg, tol=1e-50))

    v = char_lm.p_next('SELECT bb').normalize()
    print(v)
    assert set(v.trim().keys()) == {' ', 'b'}


def test_case_insensitive_char_proposal():
    grammar = r"""
    start: WS? "SELECT"i WS
    WS: /[ ]/
    """

    guide = EarleyLM(locally_normalize(LarkStuff(grammar).char_cfg(0.99)))

    assert guide.p_next('').trim().keys() == {'S', 's', ' '}
    assert guide.p_next('S').trim().keys() == {'E', 'e'}
    assert guide.p_next('s').trim().keys() == {'E', 'e'}


def test_case_insensitive_expansion():
    assert expand_case_insensitive('AND') == 'AND'
    assert expand_case_insensitive('(?i:AND)') == '[aA][nN][dD]'

    assert expand_case_insensitive('[aA][nN][dD]') == '[aA][nN][dD]'
    assert expand_case_insensitive('(?i:[aA][nN][dD])') == '[aA][nN][dD]'

    assert expand_case_insensitive('(?i:AND|OR)') == '[aA][nN][dD]|[oO][rR]'
    assert expand_case_insensitive('(?i:[aA][nN][dD]|OR)') == '[aA][nN][dD]|[oO][rR]'
    assert expand_case_insensitive('(?i:AND)|(?i:OR)') == '[aA][nN][dD]|[oO][rR]'

    assert expand_case_insensitive('(?i:[aA][nN][d)') == '[aA][nN][[dD]'
    assert expand_case_insensitive('(?i:[aA][nN][dD)') == '[aA][nN][[dD][dD]'
    assert expand_case_insensitive('(?i:[aA][nN][dE])') == '[aA][nN][[dD][eE]]'
    assert expand_case_insensitive('(?i:[aA][nN][dDE])') == '[aA][nN][[dD][dD][eE]]'

    assert expand_case_insensitive('(?i:(?i:AND))') == '[aA][nN][dD]'
    assert expand_case_insensitive('(?i:(?i:(?i:AND)))') == '[aA][nN][dD]'
    assert (
        expand_case_insensitive('(?i:(?i:(?i:AND)))(?i:(?i:AND))(?i:AND)')
        == '[aA][nN][dD][aA][nN][dD][aA][nN][dD]'
    )
    assert (
        expand_case_insensitive('(?i:(?i:(?i:AND)|(?i:OR)))') == '[aA][nN][dD]|[oO][rR]'
    )

    assert expand_case_insensitive('(?i:(AND|OR))') == '([aA][nN][dD]|[oO][rR])'
    assert expand_case_insensitive('(?i:AND|(?i:OR))') == '[aA][nN][dD]|[oO][rR]'

    assert (
        expand_case_insensitive('(?i:[a-z][A-Z][a-zA-z])') == '[a-zA-Z][a-zA-Z][a-zA-Z]'
    )
    assert (
        expand_case_insensitive('[a-z](?i:a[a-z]z)[a-z]') == '[a-z][aA][a-zA-Z][zZ][a-z]'
    )

    assert expand_case_insensitive('(?i:\s)') == '\s'
    assert expand_case_insensitive('(?i:\\\\s)') == '\\\\[sS]'

    sql_example_input = '(?:(?:(?:(?i:RIGHT)|(?i:FULL)|(?i:LEFT))(?:(?:[ \t\x0c\r\n])+(?i:OUTER))?|(?i:INNER)|(?:(?i:RIGHT)|(?i:FULL)|(?i:LEFT))|(?i:(?:(?i:OUTER))?))(?:[ \t\x0c\r\n])+)?(?i:JOIN)[ ]?'
    sql_example_output = '(?:(?:(?:[rR][iI][gG][hH][tT]|[fF][uU][lL][lL]|[lL][eE][fF][tT])(?:(?:[ \t\x0c\r\n])+[oO][uU][tT][eE][rR])?|[iI][nN][nN][eE][rR]|(?:[rR][iI][gG][hH][tT]|[fF][uU][lL][lL]|[lL][eE][fF][tT])|(?:[oO][uU][tT][eE][rR])?)(?:[ \t\x0c\r\n])+)?[jJ][oO][iI][nN][ ]?'
    assert expand_case_insensitive(sql_example_input) == sql_example_output


def test_github_issue_26_():
    # [2024-07-02 Tue] The original lark -> genparse.CFG translation of this
    # grammar had a nonterminal--terminal naming conflict.
    grammar = """
    start: x and x
    x: "b" | "a"
    and: " AND "
    """

    L = LarkStuff(grammar)

    cfg = L.char_cfg(ignore=r'')

    assert cfg.V == {'A', 'N', 'D', 'a', 'b', ' '}

    cfg.language(100).assert_equal(
        {
            ('b', ' ', 'A', 'N', 'D', ' ', 'b'): 0.25,
            ('b', ' ', 'A', 'N', 'D', ' ', 'a'): 0.25,
            ('a', ' ', 'A', 'N', 'D', ' ', 'b'): 0.25,
            ('a', ' ', 'A', 'N', 'D', ' ', 'a'): 0.25,
        }
    )

    # The original failing example is below:

    grammar = """
    start: sent
    sent: "exists " var " . " sent
    | "forall " var " . " sent
    | "( " sent " )"
    | sent " AND " sent
    | sent " OR " sent
    | expr "(" var ")"
    | expr "(" var ", " var ")"
    | expr "(" var ", " const ")"
    var: "x" | "y" | "z" | "a" | "e" | "i"
    expr: "boy" | "girl"
    const: "Bill" | "Mary"
    """

    guide = BoolCFGLM(LarkStuff(grammar).char_cfg(ignore='[ ]?'))

    guide.p_next('exists x . boy(x)').assert_equal({'▪': 1, ' ': 1})

    # The bug originally allowed 'a'
    guide.p_next('exists x . boy(x) ').assert_equal({'▪': 1, ' ': 1, 'A': 1, 'O': 1})

    guide.p_next('exists x . boy(x) a').assert_equal({})


def test_lark_ignore():
    grammar = r"""
    start: "SELECT" NAME "FROM" NAME EOS
    NAME: /[A-Za-z][A-Za-z]?[A-Za-z]?[A-Za-z]?[A-Za-z]?/
    EOS: "</s>"
    WS: /[ ]/
    %ignore WS
    """

    guide = BoolCFGLM(LarkStuff(grammar).char_cfg(0.99))

    assert guide.p_next('').keys() == {'S', ' '}
    assert guide.p_next(' ').keys() == {'S'}
    assert guide.p_next(' S').keys() == {'E'}
    assert ' ' in guide.p_next(' SELECT').keys()
    assert ' ' not in guide.p_next(' SELECT ').keys()


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
