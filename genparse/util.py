import re
import sys
import nltk
import numpy as np
from path import Path
from collections import Counter
from contextlib import contextmanager
from itertools import chain, combinations
from time import time
from IPython.display import display, SVG, Image, HTML, Latex


def bpe_wfst(S):
    """
    Create a transducer relating strings of BPE token ids to their associated strings
    """
    from genparse import Float
    from genparse.fst import FST, EPSILON
    m = FST(Float)
    START = 0
    STOP = 1
    m.add_I(0, 1)
    for i, x in S:
        m.add_arc(START, (i, EPSILON), (i, 0), 1)
        for j in range(len(x)):
            m.add_arc((i,j), (EPSILON, x[j]), (i,j+1), 1)
        m.add_arc((i,len(x)), (EPSILON, EPSILON), STOP, 1)
    m.add_F(STOP, 1)
    m.add_arc(STOP, (EPSILON, EPSILON), START, .1)   # decay
    return m.renumber


class LarkStuff:
    """
    Utility class for leveraging the lark parsing library.
    """
    def __init__(self, grammar):
        import lark
        self.raw_grammar = grammar

        builder = lark.load_grammar.GrammarBuilder()
        builder.load_grammar(grammar)
        lark_grammar = builder.build()
        terminals, rules, ignores = lark_grammar.compile(["start"], set())

        self.parser = lark.parsers.cyk.Parser(rules)
        self.instance = lark.Lark(grammar, lexer='basic', parser='cyk')
        self.lex = self.instance.lex
        self.rules = self.parser.grammar.rules
        self.terminals = terminals
        self.ignores = ignores

    def transducer(self, decay=.99, **kwargs):
        """
        XXX: Warning: There may be infelicity in the tokenization semantics as there is
        no longer a prioritized or maximum munch semantics to tokenizer.  It is
        probabilistic and the weights are set pretty arbitrarily.
        """
        from genparse import Float
        from genparse.fst import FST, EPSILON
        m = FST(Float)

        START = 0
        STOP = 1
        m.add_I(START, 1)
        m.add_F(STOP, 1)

        m.add_arc(STOP, (EPSILON, EPSILON), START, decay)

        for id, token_class in enumerate(self.terminals):
            #print('>>>', id, token_class)
            fsm = regex_to_greenery(token_class.pattern.to_regexp(), **kwargs)

            m.add_arc(START, (EPSILON, token_class.name), (id, fsm.initial), 1)

            for final_state in fsm.finals:
                m.add_arc((id, final_state), (EPSILON, EPSILON), STOP, 1)

            dead = {i for i in fsm.states if not fsm.islive(i)}
            for state in fsm.states:
                arcs = fsm.map[state]
                for input_char, next_state in arcs.items():
                    if next_state in dead: continue
                    for char in input_char.get_chars():
                        m.add_arc((id, state), (char, EPSILON), (id, next_state), 1)

        return m

    def convert(self):
        "Convert the lark grammar into a `genparse.CFG` grammar."
        from genparse import CFG, Rule
        from genparse.cfglm import Float
        terminals = [t.name for t in self.terminals]
        rules = [Rule(1, r.lhs.name, tuple(y.name for y in r.rhs)) for r in self.rules]
        lhs_count_dict = Counter([r.head for r in rules])
        rules = [normalize_rule(r, lhs_count_dict) for r in rules]
        cfg = CFG(R=Float, S="start", V=set(terminals))
        for r in rules:
            cfg.add(r.w, r.head, *r.body)
        return cfg

    def simple_tokenizer(self, text):
        """
        This is a very simple DIY tokenizer. That uses Python's `re` library.
        """
        # The regex pattern to match any of the tokens
        token_regex = '|'.join(f'(?P<{t.name}>{t.pattern.value})'
                               for t in sorted(self.terminals,
                                               key=lambda t: -t.priority))

        for match in re.finditer(token_regex, text):
            token_type = match.lastgroup
            token_value = match.group()
            if token_type not in self.ignores:
                yield token_type, token_value


def regex_to_greenery(regex, ignore = ''):
    """
    Convert `regex`, a python-like regular expression (`re`), into a `greenery`
    finite-state machine (FSM).
    """
    import greenery
    # Patch: note that greenery does not escape spaces but both the `re` and `lark` do.
    return greenery.parse(regex.replace("\\ ", " ") + ignore).to_fsm()


# Not essential; only used in a notebook to visualize individual greenery FSMs
def greenery_to_fsa(fsm):
    import fsa
    if isinstance(fsm, str): fsm = regex_to_greenery(fsm)
    m = fsa.FSA()
    m.add_start(fsm.initial)
    for final_state in fsm.finals:
        m.add_stop(final_state)
    rejection_states = [e for e in fsm.states if not fsm.islive(e)]
    for state in fsm.states:
        arcs = fsm.map[state]
        for input_char, next_state in arcs.items():
            if next_state in rejection_states:  # rejection state
                continue
            for char in input_char.get_chars():
                m.add(state, char, next_state)
    return m


def normalize_rule(rule, lhs_count_dict):
    rule.w = 1.0 / lhs_count_dict[rule.head]
    return rule


def format_table(rows, headings=None):
    def fmt(x):
        try:
            return x._repr_html_()
        except AttributeError:
            try:
                return x._repr_svg_()
            except AttributeError:
                return str(x)

    return (
        '<table>'
         + ('<tr style="font-weight: bold;">' + ''.join(f'<td>{x}</td>' for x in headings) +'</tr>' if headings else '')
         + ''.join(f'<tr>' + ''.join(f'<td>{fmt(x)}</td>' for x in row) +  ' </tr>' for row in rows)
         + '</table>'
    )


def display_table(*args, **kwargs):
    return display(HTML(format_table(*args, **kwargs)))


@contextmanager
def timeit(name, fmt='{name} ({htime})', header=None):
    """Context Manager which prints the time it took to run code block."""
    if header is not None: print(header)
    b4 = time()
    yield
    sec = time() - b4
    ht = '%.4f sec' % sec
    print(fmt.format(name=name, htime=ht, sec=sec), file=sys.stderr)


def ansi(color=None, light=None, bg=3):
    return '\x1b[%s;%s%sm' % (light, bg, color) + '%s\x1b[0m'


class colors:

    black, red, green, yellow, blue, magenta, cyan, white = \
        [ansi(c, 0) for c in range(8)]

    class light:
        black, red, green, yellow, blue, magenta, cyan, white = \
            [ansi(c, 1) for c in range(8)]

    class dark:
        black, red, green, yellow, blue, magenta, cyan, white = \
            [ansi(c, 2) for c in range(8)]

    def rgb(r,g,b): return f"\x1b[38;2;{r};{g};{b}m%s\x1b[0m"

    orange = rgb(255, 165, 0)

    purple = '\x1b[38;5;91m' + '%s' + '\x1b[0m'

    normal = '\x1b[0m%s\x1b[0m'
    bold = '\x1b[1m%s\x1b[0m'
    italic = "\x1b[3m%s\x1b[0m"
    underline = "\x1b[4m%s\x1b[0m"
    strike = "\x1b[9m%s\x1b[0m"
    #overline = lambda x: (u''.join(unicode(c) + u'\u0305' for c in unicode(x))).encode('utf-8')

    def line(n): return '─'*(n)

    def thick_line(n): return ('━'*n)

    check = green % '✔'
    xmark = dark.red % '✘'
    def mark(x): return colors.check if x else colors.xmark
