import re
import sys
import nltk
import html
import numpy as np
from path import Path
from collections import Counter
from contextlib import contextmanager
from itertools import chain, combinations
from functools import cached_property
from time import time
from IPython.display import display, SVG, Image, HTML, Latex


class hf_tokenizer:
    def __init__(self, name='gpt2'):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.pairs = [(i, self.tokenizer.decode([i]))
                      for i in range(self.tokenizer.vocab_size)]

    @cached_property
    def fst(self):
        return bpe_wfst(self.pairs)


# Warning: untested
#class hf_tokenizer_codellama:
#    def __init__(self):
#        from transformers import AutoTokenizer
#        model_name = "codellama/CodeLlama-7b-Instruct-hf"
#        self.tokenizer = AutoTokenizer.from_pretrained(
#            model_name,
#            use_fast=True,
#            prefix_token=None,
#            middle_token=None,
#            suffix_token=None,
#            eot_token=None,
#            fill_token=None,
#        )
#        self.pairs = [(i, tokenizer.decode([i])) for i in range(self.tokenizer.vocab_size)]
#
#    @cached_property
#    def fst(self):
#        return bpe_wfst(self.pairs)


def normalize(p):
    Z = sum(p[x] for x in p)
    q = p.copy()
    for x in q:
        q[x] /= Z
    return q


def bpe2term_approx(tokenizer, bpe_sequence):
    from genparse import FST, Float
    # approximate the transducer using a single canonical path;
    # UPDATE: the unpruned answer should match this - it's the uncertainty over bpe that's tricky
    c = tuple(([b], tokenizer.decode([b])) for b in bpe_sequence)
    tmp = FST.from_pairs([([], '')], Float)
    for pair in c:
        tmp = tmp * FST.from_pairs([pair], Float)
    return tmp
    # TODO: approximate this transducer by a canonical path
    #return c2t(c, None).trim.epsremove.trim


def about(m):
    print(f"states: {len(m.states)}, trim: {len(m.trim.states)}")


#def template(main, annotation):
#    return f"""\
#<div style="text-align: center; display: inline-block; font-family: Monospace; margin: 0px !important; padding: 0px !important;">
#    <div style="margin: 0px !important; padding: 0px !important; border-bottom: 1px solid #ddd;">{html.escape(str(main))}</div>
#    <div style="font-size: 8pt; color: #bbb; margin: 0px !important; padding: 0px !important;">{html.escape(str(annotation))}</div>
#</div>
#"""

def template(main, annotation):
    return f"""\
<div style="text-align: center; display: inline-block; background-color: #eee; font-family: Monospace; margin: 0px !important; padding: 0px !important;">
    <div style="display: inline-block; margin: 0px !important; padding: 0px !important;">{html.escape(str(main))}</div>
    <div style="display: inline-block; font-size: 8pt; color: #bbb; margin: 0px !important; padding: 0px !important;">/{html.escape(str(annotation))}</div>
</div>
"""

def fmt(x): return repr(x)[1:-1] if isinstance(x, str) else repr(x)

def show_grammar(cfg_t, chart=None, showzero=False):
    """Fancier pretty-printing the grammar.

    - total weight alongsize each nonterminal

    - rules are grouped by their head and how on one line separated by "|"

    - head nonterminals are grouped by SCC (i.e., mutually recursive block) and
      sort SCCs topologically (layout is top down starting from the root).

    - Grammar is trimmed to nonzero values (to bypas set `showzero=True`).

    """
    if chart is None:
        chart = cfg_t.agenda(maxiter=1000)

    def format_tokens(tokens):
        if len(tokens) == 0: return template('ε', cfg_t.R.one)
        return '<span style="padding-right: 10px;"></span>'.join(template(fmt(i), chart[i]) for i in tokens)

    lines = []

    for block in cfg_t.dependency_graph().blocks():
        if not showzero: block = [x for x in block if chart[x] != cfg_t.R.zero and cfg_t.is_nonterminal(x)]
        if not block: continue

        block_code = []
        for x in block:
            block_code.append(
                template(fmt(x), chart[x]) #+ '<br/>'
                +
                ('<div style="display: inline-block;">→ %s</div>'
                 % ' | '.join(template('', r.w) + format_tokens(r.body) for r in cfg_t.rhs[x]))
            )

#        lines.append('<div style="border-left: thick solid black; padding-left: 3px; margin-bottom: 5px;">%s</div>' % '\n'.join(block_code))
        lines.append('<div style="border-left: thick solid black; padding-left: 3px; margin-bottom: 5px;">%s</div>' % '\n'.join(block_code))

    return HTML(''.join(lines))


# TODO: should this method be factored into a method that builds the set of pairs followed
# by a call to kleene star of the transducer?
def bpe_wfst(S):
    """
    Create a transducer relating strings of BPE token ids to their associated strings
    """
    from genparse import Float, FST, EPSILON
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
    m.add_arc(STOP, (EPSILON, EPSILON), START, 1)
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
        from genparse import Float, FST, EPSILON
        m = FST(Float)

        START = 0
        STOP = 1
        m.add_I(START, 1)
        m.add_F(STOP, decay)

        m.add_arc(STOP, (EPSILON, EPSILON), START, 1)

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
                        m.add_arc((id, state), (char, EPSILON), (id, next_state), decay)

        return m

    def convert(self):
        "Convert the lark grammar into a `genparse.CFG` grammar."
        from genparse import CFG, Rule, Float
        rules = [Rule(1, r.lhs.name, tuple(y.name for y in r.rhs)) for r in self.rules]
        lhs_count = Counter([r.head for r in rules])
        cfg = CFG(R=Float, S="start", V={t.name for t in self.terminals})
        for r in rules:
            cfg.add(1/lhs_count[r.head], r.head, *r.body)
        return cfg.renumber()

    def char_cfg(self, decay):
        from genparse import CFG, Rule, Float

        cfg = self.convert()

        foo = CFG(Float, S=cfg.S, V=set())
        for r in cfg:
            foo.add(r.w, r.head, *r.body)

        for token_class in self.terminals:

            fsa = greenery_to_wfsa(token_class.pattern.to_regexp(), decay=decay,
                                   name=lambda x: (token_class.name, x))
            #display(fsa)
            G = fsa.to_cfg(S=token_class.name)
            #display(G)

            foo.V |= G.V
            for r in G:
                foo.add(r.w, r.head, *r.body)

        return foo

#    def simple_tokenizer(self, text):
#        "simple DIY prioritized tokenizer; uses Python's `re` library."
#        # The regex pattern to match any of the tokens
#        token_regex = '|'.join(f'(?P<{t.name}>{t.pattern.value})'
#                               for t in sorted(self.terminals,
#                                               key=lambda t: -t.priority))
#        for match in re.finditer(token_regex, text):
#            token_type = match.lastgroup
#            token_value = match.group()
#            if token_type not in self.ignores:
#                yield token_type, token_value


def regex_to_greenery(regex, ignore = ''):
    """
    Convert `regex`, a python-like regular expression (`re`), into a `greenery`
    finite-state machine (FSM).
    """
    import greenery
    # Patch: note that greenery does not escape spaces but both the `re` and `lark` do.
    return greenery.parse(regex.replace("\\ ", " ") + ignore).to_fsm()


# Not essential; only used in a notebook to visualize individual greenery FSMs
#def greenery_to_fsa(fsm):
#    import fsa
#    if isinstance(fsm, str): fsm = regex_to_greenery(fsm)
#    m = fsa.FSA()
#    m.add_start(fsm.initial)
#    for final_state in fsm.finals:
#        m.add_stop(final_state)
#    rejection_states = [e for e in fsm.states if not fsm.islive(e)]
#    for state in fsm.states:
#        arcs = fsm.map[state]
#        for input_char, next_state in arcs.items():
#            if next_state in rejection_states:  # rejection state
#                continue
#            for char in input_char.get_chars():
#                m.add(state, char, next_state)
#    return m


# Not essential; only used in a notebook to visualize individual greenery FSMs
def greenery_to_wfsa(fsm, decay=.99, name=lambda x: x):
    from genparse import WFSA, Float
    if isinstance(fsm, str): fsm = regex_to_greenery(fsm)
    m = WFSA(Float)
    m.add_I(name(fsm.initial), 1)

    rejection_states = [e for e in fsm.states if not fsm.islive(e)]
    for state in fsm.states:
        arcs = fsm.map[state]

        # determine this state's fan out...
        K = 0
        for input_char, next_state in arcs.items():
            if next_state in rejection_states: continue  # rejection state
            for char in input_char.get_chars():
                K += 1
        if state in fsm.finals:
            K += 1

        if K == 0: continue

        if state in fsm.finals:
            m.add_F(name(state), decay / K)

        for input_char, next_state in arcs.items():
            if next_state in rejection_states: continue  # rejection state
            for char in input_char.get_chars():
                m.add_arc(name(state), char, name(next_state), decay / K)

    return m


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
