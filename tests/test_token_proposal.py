import numpy as np
from arsenal import timeit

from genparse.util import set_seed
from genparse.cfglm import add_EOS, locally_normalize, BoolMaskCFGLM
from genparse.experimental.earley import EarleyLM
from genparse.lm import make_mock_llm, MockLLM
from genparse.proposal import TokenProposal
from genparse.util import LarkStuff
from genparse.proposal.util import (
    mock_token_proposal,
    assert_proper_weighting,
    assert_unbiased_Z,
)

# TODO: test equivalence of `traverse_trie` and `traverse_naive`.
# def traverse_naive(self, context):
#    for x in self.words:
#        p_x = self.guide.pfg(context + x)  # prefix weight of context + x
#        if p_x == 0: continue
#        yield (context + x, p_x)


def test_basic_aligned_model_iql_small():
    set_seed(0)

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


def test_normalizing_constant_unbiased():
    """
    The expected importance weight should provide an unbiased estimate of the normalizing constant.
    That is, we expect E_{(x,S) ~ q(x,S)}[w(x,S)] = Σ_x p(x).
    """
    set_seed(0)

    V = {
        '▪',
        ' ',
        '  ',
        ' W',
        ' O',
        ' S',
        ' s',
        ' WHE',
        ' SEL',
        ' ORD',
        ' sta',
        ' WHER',
        ' SELE',
        ' ORDE',
        ' stat',
        ' stad',
        ' SELECT',
        ' WHERE',
        ' ORDER',
        ' state',
        ' stadium',
    }

    grammar = r"""
            start: WS? "SELECT" WS select_expr WS "FROM" WS from_expr [WS "WHERE" WS bool_condition] [WS "GROUP BY" WS var_list] [WS "ORDER BY" WS orderby_expr] WS EOS
            EOS: "▪"
            select_expr: STAR | select_list
            bool_condition: bool_expr | "(" bool_condition WS "AND" WS bool_condition ")" | "(" bool_condition WS "OR" WS bool_condition ")"
            bool_expr: var "=" value | var ">" value | var "<" value
            from_expr: "data"
            orderby_expr: var_list WS "ASC" | var_list WS "DESC"
            select_list: select_var ("," WS select_var)*
            var_list: var ("," WS var)*
            select_var: var | "AVG(" var ")" | "MEDIAN(" var ")" | "COUNT(" var ")"
            var: "state" | "stadium"
            value: NUMBER | "'red'"
            STAR: "*"
            NUMBER: /\d+/
            WS: /[ ]/
     """

    proposal = mock_token_proposal(V=V, guide_spec=grammar, K=10, uniform=True)

    prompt = ''
    context = ' '

    assert_unbiased_Z(prompt, context, proposal, tol=1e-8)

    prompt = ''
    context = ' SELECT'

    assert_unbiased_Z(prompt, context, proposal, tol=1e-8)

    prompt = ''
    context = ' SELECT * FROM data'

    assert_unbiased_Z(prompt, context, proposal, tol=1e-8)


def test_proper_weighting():
    r"""
    A particle (x,w) is *properly weighted* for unnormalized density p' if, for any function f,

        E_{(x,w) ~ \tilde{q}}[f(x)w] = Σ_x p'(x) f(x)

    where Z normalizes p'. In our case, we have that

        E_{(x,w) ~ \tilde{q}}[f(x)w] = E_{(x,S) ~ q}[f(x)w(x,S)]

    Thus, we expect

        E_{(x,S) ~ q}[f(x)w(x,S)] = Σ_x p'(x) f(x)

    for the local product of experts distributions. We test this for f(x) = δ(x', x) for all x' ∈ V.
    """
    set_seed(0)

    V = {' ', ' a', ' b', '▪'}

    grammar = r"""
        start: WS x EOS
        EOS: "▪"
        x: "a" | "b" | "ab"
        WS: /[ ]/
    """

    proposal = mock_token_proposal(V=V, guide_spec=grammar, K=2, uniform=True)

    prompt = ''
    context = ''

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)

    V = {
        '▪',
        ' ',
        '  ',
        ' W',
        ' O',
        ' S',
        ' s',
        ' WHE',
        ' SEL',
        ' ORD',
        ' sta',
        ' WHER',
        ' SELE',
        ' ORDE',
        ' stat',
        ' stad',
        ' SELECT',
        ' WHERE',
        ' ORDER',
        ' state',
        ' stadium',
    }

    grammar = r"""
        start: WS? "SELECT" WS select_expr WS "FROM" WS from_expr [WS "WHERE" WS bool_condition] [WS "GROUP BY" WS var_list] [WS "ORDER BY" WS orderby_expr] WS EOS
        EOS: "▪"
        select_expr: STAR | select_list
        bool_condition: bool_expr | "(" bool_condition WS "AND" WS bool_condition ")" | "(" bool_condition WS "OR" WS bool_condition ")"
        bool_expr: var "=" value | var ">" value | var "<" value
        from_expr: "data"
        orderby_expr: var_list WS "ASC" | var_list WS "DESC"
        select_list: select_var ("," WS select_var)*
        var_list: var ("," WS var)*
        select_var: var | "AVG(" var ")" | "MEDIAN(" var ")" | "COUNT(" var ")"
        var: "state" | "stadium"
        value: NUMBER | "'red'"
        STAR: "*"
        NUMBER: /\d+/
        WS: /[ ]/
    """

    proposal = mock_token_proposal(V=V, guide_spec=grammar, K=10, uniform=False)

    prompt = ''
    context = ' SELECT'

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)

    prompt = ''
    context = ' SELECT * FROM data'

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)

    #######################
    # Probabilistic guide #
    #######################

    pcfg = EarleyLM.from_string(
        """

        1: S -> a
        1: S -> a a
        2: S -> a a a

        """
    )

    V = {'a', 'aa', 'aaa', '▪'}

    proposal = mock_token_proposal(V=V, guide_spec=pcfg, K=2, uniform=True)

    prompt = ''
    context = ''

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)

    prompt = ''
    context = 'a'

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)


# TODO: fix this error!
def todo_github_issue_15_wildcard_divide_by_zero():
    guide = BoolMaskCFGLM.from_string(
        """

        1: S -> a
        1: S -> a a
        1: S -> a a a

        """
    )

    V = ['a', 'aa', 'aaa', '▪']

    llm = MockLLM(V=V, eos='▪', _p=np.array([0, 0, 1, 0]))

    proposal = TokenProposal(llm=llm, guide=guide, K=1)

    context = 'aa'

    assert_proper_weighting('', context, proposal, tol=1e-8)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
