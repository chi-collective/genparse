import lark
import string
import interegular
from interegular.fsm import anything_else

import arsenal
import warnings
from collections import Counter

from genparse import WFSA, Float
from genparse.cfg import CFG, Rule


class LarkStuff:
    """Utility class for leveraging the lark as a front-end syntax for specifying
    grammars.

    Warning: There may be infelicity in the tokenization semantics as there is
    no longer a prioritized or maximum-munch semantics to the tokenizer when we
    encode it into the grammar.

    NOTE: In conversion from lark to genparse, there are numerous features that
    need to be handled with care.

    * Notably, the `ignore` directive in lark is supported by concatenating
      existing terminal class regexes with an optional prefix containing the
      ignore terms. The semantics of this are equivalent, but the implementation
      is not.

    * When lark compiles terminal class regexes to python re syntax, not all
      features are supported by greenery.

      - Our implementations of `.` and `^` are in terms of negated character
        classes, and require special handling.  In our conversion, we consider
        negation with respect to a superset defined by `string.printable`. There
        may be other cases we have not yet encountered, so it is important to
        verify that conversions are correct when incorporating new grammars. We
        expect edge cases with lookahead and lookbehind assertions to be
        particularly problematic.

    TODO: update now that greenery has been replaced by interegular

    """

    def __init__(self, grammar, cnf=False):
        self.raw_grammar = grammar

        builder = lark.load_grammar.GrammarBuilder()
        builder.load_grammar(grammar)
        lark_grammar = builder.build()

        if not any(
            rule.value == 'start'
            for rule in lark_grammar.rule_defs[0]
            if isinstance(rule, lark.lexer.Token)
        ):
            raise ValueError('Grammar must define a `start` rule')

        terminals, rules, ignores = lark_grammar.compile(['start'], set())

        if cnf:
            self.parser = lark.parsers.cyk.Parser(rules)
            # self.instance = lark.Lark(grammar, lexer='basic', parser='cyk')
            # self.lex = self.instance.lex
            self.rules = self.parser.grammar.rules

        else:
            # self.parser = lark.parsers.earley.Parser(rules)
            # self.instance = lark.Lark(grammar, parser='earley')
            # self.lex = self.instance.lex
            self.rules = rules

        self.terminals = terminals
        self.ignore_terms = ignores
        self.ignore_regex = f'(?:{"|".join([t.pattern.to_regexp() for t in self.terminals if t.name in ignores])})?'

    def convert(self):
        "Convert the lark grammar into a `genparse.CFG` grammar."

        try:
            rules = [
                Rule(1, r.lhs.name, tuple(y.name for y in r.rhs)) for r in self.rules
            ]
        except AttributeError:
            rules = [
                Rule(1, r.origin.name, tuple(y.name for y in r.expansion))
                for r in self.rules
            ]

        lhs_count = Counter([r.head for r in rules])
        cfg = CFG(R=Float, S='start', V={t.name for t in self.terminals})
        for r in rules:
            cfg.add(1 / lhs_count[r.head], r.head, *r.body)
        return cfg.renumber()

    def char_cfg(self, decay=None, delimiter='', charset='core'):
        if decay is not None:
            warnings.warn('Option `decay` is deprecated')

        if delimiter:
            warnings.warn(
                'Use of delimiter enforced between terminals. If delimiter is not a strict subset of `%ignore`, generated strings will deviate from original grammar.'
            )

        cfg = self.convert()

        # rename all of the internals to avoid naming conflicts.
        f = arsenal.Integerizer()

        foo = CFG(Float, S=f(cfg.S), V=set())
        for r in cfg:
            foo.add(r.w, f(r.head), *(f(y) for y in r.body))

        for token_class in self.terminals:
            if token_class.name in self.ignore_terms:
                continue
            regex = self.ignore_regex + token_class.pattern.to_regexp() + delimiter

            fsa = interegular_to_wfsa(
                regex,
                name=lambda x, t=token_class.name: f((t, x)),
                charset=charset,
            )
            # display(fsa)
            G = fsa.to_cfg(S=f(token_class.name))

            foo.V |= G.V
            for r in G:
                foo.add(r.w, r.head, *r.body)

        assert len(foo.N & foo.V) == 0

        return foo


def interegular_to_wfsa(pattern, name=lambda x: x, charset='core'):
    if charset == 'core':
        charset = set(string.printable)
    else:
        # TODO: implement other charsets
        raise NotImplementedError(f'charset {charset} not implemented')

    # Compile the regex pattern to an FSM
    fsm = interegular.parse_pattern(pattern).to_fsm()

    # if anything_else in fsm.alphabet:
    #    print(arsenal.colors.orange % 'ALPHABET:', set(fsm.alphabet))
    #    print(arsenal.colors.orange % 'ANYTHING ELSE:', charset - set(fsm.alphabet))

    def expand_alphabet(a):
        if anything_else in fsm.alphabet.by_transition[a]:
            assert fsm.alphabet.by_transition[a] == [anything_else]
            return charset - set(fsm.alphabet)
        else:
            return fsm.alphabet.by_transition[a]

    m = WFSA(Float)
    m.add_I(name(fsm.initial), 1)

    rejection_states = [e for e in fsm.states if not fsm.islive(e)]
    for i in fsm.states:
        # determine this state's fan out
        K = 0
        for a, j in fsm.map[i].items():
            # print(f'{i} --{a}/{fsm.alphabet.by_transition[a]}--> {j}')
            if j in rejection_states:
                continue
            for A in expand_alphabet(a):
                assert isinstance(A, str)
                if len(A) != 1:
                    warnings.warn(
                        f'Excluding multi-character arc {A!r} in pattern {pattern!r} (possibly a result of case insensitivity of arcs {expand_alphabet(a)})'
                    )
                    continue
                K += 1
        if i in fsm.finals:
            K += 1
        if K == 0:
            continue
        if i in fsm.finals:
            m.add_F(name(i), 1 / K)
        for a, j in fsm.map[i].items():
            if j in rejection_states:
                continue
            for A in expand_alphabet(a):
                m.add_arc(name(i), A, name(j), 1 / K)

    return m
