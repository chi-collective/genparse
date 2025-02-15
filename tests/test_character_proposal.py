import numpy as np
from arsenal import colors, timeit

from genparse import Float, EarleyLM
from genparse.proposal import CharacterProposal
from genparse.util import set_seed, lark_guide, load_model_by_name

from genparse.proposal.util import (
    mock_character_proposal,
    assert_proper_weighting,
    assert_unbiased_Z,
)


def test_timothy():
    set_seed(0)

    guide = lark_guide(r""" start: /[ ]*Tim(othy)?[ ](Fabbri[ ])?Vieira\./""")

    llm = load_model_by_name('gpt2')
    prompt = llm.encode_prompt('Hello my name is')

    proposal = CharacterProposal(llm=llm, guide=guide)
    with timeit('sample'):
        for _ in range(10):
            print(colors.line(80))
            x, q, w = proposal.sample(prompt, max_tokens=50)
            print(
                f'{np.log(q):.2f}\t{np.log(w):.2f}\t',
                (colors.light.cyan % '[')
                + (colors.light.cyan % '|').join(x)
                + (colors.light.cyan % ']'),
            )


def todo_chomsky():
    set_seed(0)

    guide = lark_guide(
        r"""

        start: /Noam[ ]Chomsky[ ]famously[ ]wrote,[ ]"/ expr /\."/

        //expr: /[A-Za-z0-9,; ]+/
        expr: /[Tt]ime[ ]flies[ ]like[ ]an[ ]arrow/
        | /[iI][ ]like[ ]to[ ]dance/
        | /[cC]olorless[ ]green[ ]ideas[ ]sleep[ ]furiously/

        """
    )

    # TODO: we can improve this model considerably by encoding the max length
    # into the CFG as it will push backward the constraint that `."` needs to be
    # generated.  Currently, most of the samples generated have weight zero
    # because of the '."' technicality!  Encoding the constraint exactly might
    # be a bit of a challenge as the set of tokens of length <= T is an
    # expensive constraint to encoded exactly.  But, we can approximate with
    # something simpler like the number of white spaces in many settings.

    # XXX: we are using the boolean CFG instead of the PCFG; the PCFG is running
    # into numerical underflow.  We need to use the log-semiring or a rescaling
    # trick in the Earley parser.

    llm = load_model_by_name('gpt2')
    prompt = (llm.eos,)

    W = Float.chart()

    proposal = CharacterProposal(llm=llm, guide=guide)
    for _ in range(10):
        print('----------------------------------')
        with timeit('sample'):
            ys, _, w = proposal.sample(prompt, verbosity=1)

        W[ys] += w

        print(colors.light.yellow % 'sample:', ys)

        # print(W.normalize())
        print(W.project(''.join).normalize())


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
        ' ORD',
        ' SEL',
        ' sta',
        ' WHER',
        ' SELE',
        ' ORDE',
        ' stat',
        ' stad',
        ' SELECT',
        ' WHERE',
        ' FROM',
        ' data',
        ' ORDER',
        ' state',
        ' stadium',
        ' *',
        'd',
        't',
        'a',
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
            SPACE: " "
            %ignore SPACE
     """

    proposal = mock_character_proposal(V=V, guide_spec=grammar, uniform=True)

    prompt = ()
    context = ()

    assert_unbiased_Z(prompt, context, proposal, tol=1e-8)

    prompt = ()
    context = (' SELECT',)

    assert_unbiased_Z(prompt, context, proposal, tol=1e-8)

    prompt = ()
    context = (' SELECT', ' *', ' FROM', ' data')

    assert_unbiased_Z(prompt, context, proposal, tol=1e-8)


def test_proper_weighting():
    """
    A particle (x,w) is *properly weighted* for unnormalized density p' if, for any function f,

        E_{(x,w) ~ \\tilde{q}}[f(x)w] = Σ_x p'(x) f(x)

    where Z normalizes p'. In our case, we have that

        E_{(x,w) ~ \\tilde{q}}[f(x)w] = E_{(x,S) ~ q}[f(x)w(x,S)]

    Thus, we expect

        E_{(x,S) ~ q}[f(x)w(x,S)] = Σ_x p'(x) f(x)

    for the local product of experts distributions. We test this for f(x) = δ(x', x) for all x' ∈ V.
    """
    set_seed(0)

    #################
    # Boolean guide #
    #################

    V = {' ', ' a', ' b', '▪'}

    grammar = r"""
        start: WS x EOS
        EOS: "▪"
        x: "a" | "b" | "ab"
        WS: /[ ]/
    """

    proposal = mock_character_proposal(V=V, uniform=True, guide_spec=grammar)

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
        ' ORD',
        ' SEL',
        ' sta',
        ' WHER',
        ' ORDE',
        ' SELE',
        ' stat',
        ' stad',
        ' SELECT',
        ' WHERE',
        ' ORDER',
        ' state',
        ' data',
        ' FROM',
        ' *',
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

    proposal = mock_character_proposal(V=V, guide_spec=grammar, uniform=True)

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

    proposal = mock_character_proposal(V=V, guide_spec=pcfg, uniform=True)

    prompt = ()
    context = ()

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)

    prompt = ()
    context = ('a',)

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
