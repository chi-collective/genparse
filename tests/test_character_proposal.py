from arsenal import colors, timeit

from genparse.parse.earley import EarleyLM
from genparse.cfglm import BoolMaskCFGLM, locally_normalize
from genparse.proposal import CharacterProposal
from genparse.semiring import Float
from genparse.util import LarkStuff, set_seed, load_model_by_name

from genparse.proposal.util import (
    mock_character_proposal,
    assert_proper_weighting,
    assert_unbiased_Z,
)


def test_timothy():
    set_seed(0)

    pcfg = EarleyLM(
        locally_normalize(
            LarkStuff(r""" start: /[ ]*Tim(othy)?[ ](Fabbri[ ])?Vieira\./""").char_cfg(
                0.99
            ),
            tol=1e-100,
        )
    )

    prompt = 'Hello my name is'
    llm = load_model_by_name('gpt2')

    proposal = CharacterProposal(llm=llm, guide=pcfg)
    W = Float.chart()
    for _ in range(10):
        print('----------------------------------')
        with timeit('sample'):
            ys, _, w = proposal.sample(prompt, verbosity=1, max_tokens=50)

        W[ys] += w

        print(colors.light.yellow % 'sample:', ys)

        print(W.normalize())


def todo_chomsky():
    set_seed(0)

    pcfg = EarleyLM(
        locally_normalize(
            LarkStuff(
                r"""

                start: /Noam[ ]Chomsky[ ]famously[ ]wrote,[ ]"/ expr /\."/

                //expr: /[A-Za-z0-9,; ]+/
                expr: /[Tt]ime[ ]flies[ ]like[ ]an[ ]arrow/
                  | /[iI][ ]like[ ]to[ ]dance/
                  | /[cC]olorless[ ]green[ ]ideas[ ]sleep[ ]furiously/

                """
            ).char_cfg(0.9999),
            tol=1e-300,
        )
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
    pcfg = BoolMaskCFGLM(pcfg.cfg)

    # print(''.join(pcfg.sample()))

    #    from genparse.semiring import Log
    #    tmp = pcfg.cfg.spawn(R = Log)
    #    for r in pcfg.cfg:
    #        tmp.add(Log(np.log(r.w)), r.head, *r.body)
    #    lpcfg = EarleyLM(tmp)

    #    x = 'Noam Chomsky famously wrote, "One of the most outrageous things about Modernity has always been muckraking in human nature; it has deceptively distorted the way in which one views human rights by making dece'
    #    lp = lpcfg.p_next(x)
    #    pp = pcfg.p_next(x)

    prompt = ' '
    llm = load_model_by_name('gpt2')

    W = Float.chart()

    proposal = CharacterProposal(llm=llm, guide=pcfg)
    for _ in range(10):
        print('----------------------------------')
        with timeit('sample'):
            ys, _, w = proposal.sample(prompt, verbosity=1)

        W[ys] += w

        print(colors.light.yellow % 'sample:', ys)

        print(W.normalize())


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

    proposal = mock_character_proposal(V=V, guide_spec=grammar, uniform=True)

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

    proposal = mock_character_proposal(V=V, guide_spec=pcfg, uniform=True)

    prompt = ''
    context = ''

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)

    prompt = ''
    context = 'a'

    assert_proper_weighting(prompt, context, proposal, tol=1e-8)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
