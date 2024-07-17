import numpy as np
from arsenal import colors, timeit
from arsenal.maths.combinatorics import permute

from genparse import locally_normalize, BoolCFGLM, EarleyLM, MockLLM, Float
from genparse.segmentation import prefixes
from genparse.proposal import TokenProposal
from genparse.util import set_seed, lark_guide, LarkStuff, load_model_by_name
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


def test_timothy():
    set_seed(0)

    guide = lark_guide(
        r"""
        start: /[ ]*Tim(othy)?[ ](Fabbri[ ])?Vieira\./
        """
    )

    llm = load_model_by_name('gpt2')
    prompt = llm.encode_prompt('Hello my name is')

    guide.V |= {w for word in llm.V for w in word}

    proposal = TokenProposal(llm=llm, guide=guide, K=10)
    with timeit('sample'):
        for _ in range(10):
            print(colors.line(80))
            x, q = proposal.sample(prompt, max_tokens=50)
            print(
                f'{np.log(q):.2f}\t',
                (colors.light.cyan % '[')
                + (colors.light.cyan % '|').join(x)
                + (colors.light.cyan % ']'),
            )


def test_top_K():
    from arsenal.iterextras import take

    set_seed(0)

    guide = lark_guide(
        r"""
        start: /[ ]*Tim(othy)?[ ](Fabbri[ ])?Vieira\./
        """
    )

    llm = load_model_by_name('gpt2')

    guide.V |= {w for word in llm.V for w in word}

    proposal = TokenProposal(llm=llm, guide=guide, K=None)

    context = ('Tim',)
    p_llm = llm.p_next(context)

    # Test that `traverse_trie`'s token's are returned in order of largest to
    # smallest probability
    prev = 1
    for y, p_y in proposal.traverse_trie(context, p_llm):
        assert p_y <= prev
        prev = p_y
        assert np.allclose(p_y, p_llm[y] * guide.p_next_seq(''.join(context), y))
        print(prev, '\t', repr(y))


def test_basic_aligned_model_iql_small():
    set_seed(0)

    llm = load_model_by_name('mock-gpt2')

    # the base character-level CFG language model
    guide = EarleyLM(
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
            ).char_cfg(),
            tol=1e-100,
        ).trim()
    )

    proposal = TokenProposal(guide=guide, llm=llm)

    #    proposal._prompt = ()

    #   p = proposal._p_next(())
    #   print(p)
    #   assert p.keys() == {'S', 'SE', 'SELECT'}

    #    p = proposal._p_next(('SELECT', ' *', ' FROM', ' data'))
    #    print(p)
    #    assert p.keys() == {
    #        ' ',
    #        ' <',
    #        ' </',
    #        ' G',
    #        ' W',
    #        ' O',
    #        ' GR',
    #        ' WH',
    #        ' OR',
    #        ' GROUP',
    #        ' WHERE',
    #        ' ORDER',
    #    }
    #
    #    p = proposal._p_next(('SELECT', ' age', ' FROM', ' data'))
    #    print(p)
    #    assert p.keys() == {
    #        ' ',
    #        ' <',
    #        ' </',
    #        ' G',
    #        ' W',
    #        ' O',
    #        ' GR',
    #        ' WH',
    #        ' OR',
    #        ' GROUP',
    #        ' WHERE',
    #        ' ORDER',
    #    }

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
        ' *',
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
        ' data',
        ' FROM',
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

    prompt = ()
    context = (' ',)

    assert_unbiased_Z(prompt, context, proposal, tol=1e-8)

    prompt = ()
    context = (' SELECT',)

    assert_unbiased_Z(prompt, context, proposal, tol=1e-8)

    prompt = ()
    context = (' SELECT', ' *', ' FROM', ' data')

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

    prompt = ()
    context = ()

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
        ' FROM',
        ' *',
        ' data',
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

    prompt = ()
    context = (' SELECT',)

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)

    prompt = ()
    context = (' SELECT', ' *', ' FROM', ' data')

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

    prompt = ()
    context = ()

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)

    prompt = ()
    context = ('a',)

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)


def test_no_valid_tokens():
    guide = BoolCFGLM.from_string(
        """

        1: S -> a
        1: S -> a a
        1: S -> a a a

        """
    )

    V = ['a', 'aa', 'aaa', '▪']

    llm = MockLLM(V=V, eos='▪', _p=np.array([0, 0, 1, 0]))

    proposal = TokenProposal(llm=llm, guide=guide, K=1)

    context = ('aa',)

    assert_proper_weighting((), context, proposal, tol=1e-8)

    context = ('a', 'a')

    assert_proper_weighting((), context, proposal, tol=1e-8)


def test_no_valid_wildcard_tokens():
    guide = BoolCFGLM.from_string(
        """

        1: S -> a
        1: S -> a a
        1: S -> a a a

        """
    )

    V = ['a', 'aa', 'aaa', '▪']

    llm = MockLLM(V=V, eos='▪', _p=np.array([0, 1, 0, 0]))

    proposal = TokenProposal(llm=llm, guide=guide, K=2)

    context = ('a',)

    assert_proper_weighting((), context, proposal, tol=1e-8)


def test_issue_25():
    guide = EarleyLM.from_string(
        """
        1: S -> a
        1: S -> a a
        1: S -> a a a
        """
    )

    V = ['a', 'aa', 'aaa', '▪']

    for context in prefixes(tuple('aaa▪')):
        print(colors.cyan % 'context:', context)

        for perm in permute([1, 2, 3, 4]):
            llm = MockLLM(V=V, eos='▪', _p=np.array(list(perm)) / sum(perm))

            p_llm = llm.p_next(context)
            p_guide = Float.chart(
                {k: guide.p_next_seq(''.join(context), k) for k in p_llm.keys()}
            )

            want = p_llm.materialize() * p_guide

            proposal = TokenProposal(llm=llm, guide=guide, K=None)

            # Part 1: check that the results are returned in sorted order (from highest to lowest)
            prev = 1
            have = Float.chart()
            for x in proposal.traverse_trie(context, p_llm):
                assert prev >= x[1], 'error: out-of-order enumeration!'
                prev = x[1]
                have[x[0]] = x[1]

            # Part 2: check that we return the correct values for the distributions
            have.assert_equal(want)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
