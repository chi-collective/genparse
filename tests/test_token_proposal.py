from arsenal import timeit

from genparse.cfglm import add_EOS, locally_normalize
from genparse.experimental.earley import EarleyLM
from genparse.lm import make_mock_llm
from genparse.proposal import TokenProposal
from genparse.util import LarkStuff

# TODO: test equivalence of `traverse_trie` and `traverse_naive`.
# def traverse_naive(self, context):
#    for x in self.words:
#        p_x = self.guide.pfg(context + x)  # prefix weight of context + x
#        if p_x == 0: continue
#        yield (context + x, p_x)


def test_basic_aligned_model_iql_small():
    import random

    import numpy as np

    np.random.seed(0)
    random.seed(0)

    llm = make_mock_llm()

    # the base character-level CFG language model
    cfg = add_EOS(
        locally_normalize(
            LarkStuff(
                r"""
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
            ).char_cfg(0.9),
            tol=1e-100,
        ).trim()
    )

    guide = EarleyLM(cfg)
    # guide = CFGLM(cfg)

    proposal = TokenProposal(guide=guide, llm=llm)

    proposal._prompt = ''

    with timeit('took'):
        p = proposal._p_next('')
        print(p)
        assert p.keys() == {'S', 'SE', 'SELECT'}

        p = proposal._p_next('SELECT * FROM data')
        print(p)
        assert p.keys() == {
            ' ',
            ' <',
            ' </',
            ' G',
            ' W',
            ' O',
            ' GR',
            ' WH',
            ' OR',
            ' GROUP',
            ' WHERE',
            ' ORDER',
        }

        p = proposal._p_next('SELECT age FROM data')
        print(p)
        assert p.keys() == {
            ' ',
            ' <',
            ' </',
            ' G',
            ' W',
            ' O',
            ' GR',
            ' WH',
            ' OR',
            ' GROUP',
            ' WHERE',
            ' ORDER',
        }

    print(proposal.sample())


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
