import regex
from lark import Lark
from lark.exceptions import UnexpectedCharacters, UnexpectedToken

from genparse.lm import LM
from genparse.semiring import Float


class LarkGuide(LM):
    def __init__(self, llm, grammar, start, allow_ws=True):
        # self.grammar = grammar
        # self.start = start
        self.llm = llm
        self.parser = Lark(grammar, start=start, parser='lalr', regex=True)
        self.terminal2pattern = {
            k: v.pattern.to_regexp() for k, v in self.parser._terminals_dict.items()
        }
        self.terminal2pattern['$END'] = ''
        self.allow_ws = allow_ws

        self._cache = {}
        super().__init__(V=llm.V, eos=llm.eos)

    def __call__(self, x):
        if self.allow_ws:
            return self.next_token_pattern(x) == regex.compile('|\\s+')
        else:
            return self.next_token_pattern(x) == regex.compile('')

    # TODO: It's probably much more efficient to sample a token from the LLM and
    # then check if it is legal rather than to eagerly compute the entire mask.
    # If the token is illegal, we can zero out its probability and renormalize.
    # I suppose that has the issue that we don't know the exact sampling
    # probability without doing the eager computation -- this is basically the
    # same issue as our token vs character based proposal distrbution.
    def p_next(self, context):
        pattern = self.next_token_pattern(context)
        p = Float.chart()
        for v in self.V:
            if pattern.fullmatch(v, partial=True):
                p[v] = 1
        if pattern.match(''):
            p[self.llm.eos] = 1
        return p

    def next_token_pattern(self, prefix):
        pattern = self._cache.get(prefix)
        if pattern is None:
            pattern = self._next_token_pattern(prefix)
            self._cache[prefix] = pattern
        return pattern

    def _next_token_pattern(self, prefix):
        p = self.parser.parse_interactive(prefix)
        s = p.parser_state

        try:
            for token in s.lexer.lex(s):
                s.feed_token(token)
        except (UnexpectedCharacters, UnexpectedToken):
            pass

        valid_tokens = p.accepts()

        # Get the regex for the valid tokens
        valid_regex = [self.terminal2pattern[t] for t in valid_tokens]
        if valid_regex and self.allow_ws:
            valid_regex.append(r'\s+')

        pattern = regex.compile('|'.join(valid_regex))

        return pattern


def test_basics():
    json_grammar = r"""
    ?value: dict
        | list
        | string
        | SIGNED_NUMBER      -> number
        | "true"             -> true
        | "false"            -> false
        | "null"             -> null

    list : "[" [value ("," value)*] "]"

    dict : "{" [pair ("," pair)*] "}"
    pair : string ":" value

    string : /"[^"]*"/

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """

    from genparse.util import make_mock_llm

    llm = make_mock_llm()

    guide = LarkGuide(llm, json_grammar, 'value')
    text = '{"8W{0sxM{{}]]vpEC4|i;]V@Jg_#P^j\n?k%noXNt\2#2]a8a\PJru]/`M6gaqb@EhFx"'

    text = '{"a": 1}'

    print(guide.p_next(text))

    print(guide.next_token_pattern(text))

    print(guide(text))


#    valid_regexes = json_comp_engine.complete(text)
#    empty = regex.compile('')
#    print(valid_regexes)
#    print(valid_regexes == empty)
#    print(valid_regexes.fullmatch('"abc', partial=True))
#    # end_token = Token.new_borrow_pos('$END', '', token) if token else Token('$END', '', 0, 1, 1)
#    # interactive_parser.parser_state.feed_token(end_token, True)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
