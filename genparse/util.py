import html
from collections import Counter
from functools import cached_property

from IPython.display import HTML, display

from genparse.tokenization import decode_tokenizer_vocab

import numpy as np
import random
import torch
import transformers


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class hf_tokenizer:
    def __init__(self, name='gpt2', **kwargs):
        from transformers import AutoTokenizer

        if name == 'codellama':
            name = 'codellama/CodeLlama-7b-Instruct-hf'
            _kwargs = dict(
                use_fast=True,
                prefix_token=None,
                middle_token=None,
                suffix_token=None,
                eot_token=None,
                fill_token=None,
            )
            _kwargs.update(**kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)

        # there are many ways to extract the string representations of each
        # token from the HF tokenizers.

        # tokenizer.convert_ids_to_tokens
        self.decode = decode_tokenizer_vocab(self.tokenizer)
        # string <-> token id mappings
        # self.str2int = dict(self.tokenizer.vocab)
        # self.int2str = {v: k for k, v in self.tokenizer.vocab.items()}

        self.pairs = list(enumerate(self.decode))
        self.eos = self.tokenizer.eos_token

    @cached_property
    def fst(self):
        from genparse.segmentation import bpe_wfst

        return bpe_wfst(self.pairs)


# def normalize(p):
#    Z = sum(p[x] for x in p)
#    q = p.copy()
#    for x in q:
#        q[x] /= Z
#    return q


def bpe2term_approx(tokenizer, bpe_sequence):
    from genparse import FST, Float

    # approximate the transducer using a single canonical path;
    # UPDATE: the unpruned answer should match this - it's the uncertainty over bpe that's tricky
    c = tuple(
        ([b], tokenizer.convert_ids_to_tokens(b).replace('Ġ', ' ')) for b in bpe_sequence
    )
    tmp = FST.from_pairs([([], '')], Float)
    for pair in c:
        tmp = tmp * FST.from_pairs([pair], Float)
    return tmp
    # TODO: approximate this transducer by a canonical path
    # return c2t(c, None).trim.epsremove.trim


def about(m):
    print(f'states: {len(m.states)}, trim: {len(m.trim.states)}')


# def template(main, annotation):
#    return f"""\
# <div style="text-align: center; display: inline-block; font-family: Monospace; margin: 0px !important; padding: 0px !important;">
#    <div style="margin: 0px !important; padding: 0px !important; border-bottom: 1px solid #ddd;">{html.escape(str(main))}</div>
#    <div style="font-size: 8pt; color: #bbb; margin: 0px !important; padding: 0px !important;">{html.escape(str(annotation))}</div>
# </div>
# """


def template(main, annotation):
    return f"""\
<div style="text-align: center; display: inline-block; background-color: #eee; font-family: Monospace; margin: 0px !important; padding: 0px !important;">
    <div style="display: inline-block; margin: 0px !important; padding: 0px !important;">{html.escape(str(main))}</div>
    <div style="display: inline-block; font-size: 8pt; color: #bbb; margin: 0px !important; padding: 0px !important;">/{html.escape(str(annotation))}</div>
</div>
"""


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

    def fmt(x):
        return repr(x)[1:-1] if isinstance(x, str) else repr(x)

    def format_tokens(tokens):
        if len(tokens) == 0:
            return template('ε', cfg_t.R.one)
        return '<span style="padding-right: 10px;"></span>'.join(
            template(fmt(i), chart[i]) for i in tokens
        )

    lines = []

    for block in cfg_t.dependency_graph().blocks:
        if not showzero:
            block = [
                x for x in block if chart[x] != cfg_t.R.zero and cfg_t.is_nonterminal(x)
            ]
        if not block:
            continue

        block_code = []
        for x in block:
            block_code.append(
                template(fmt(x), chart[x])  # + '<br/>'
                + (
                    '<div style="display: inline-block;">→ %s</div>'
                    % ' | '.join(
                        template('', r.w) + format_tokens(r.body) for r in cfg_t.rhs[x]
                    )
                )
            )

        #        lines.append('<div style="border-left: thick solid black; padding-left: 3px; margin-bottom: 5px;">%s</div>' % '\n'.join(block_code))
        lines.append(
            '<div style="border-left: thick solid black; padding-left: 3px; margin-bottom: 5px;">%s</div>'
            % '\n'.join(block_code)
        )

    return HTML(''.join(lines))


class LarkStuff:
    """
    Utility class for leveraging the lark parsing library.
    """

    def __init__(self, grammar, cnf=False):
        import lark

        self.raw_grammar = grammar

        builder = lark.load_grammar.GrammarBuilder()
        builder.load_grammar(grammar)
        lark_grammar = builder.build()
        terminals, rules, ignores = lark_grammar.compile(['start'], set())

        if cnf:
            self.parser = lark.parsers.cyk.Parser(rules)
            self.instance = lark.Lark(grammar, lexer='basic', parser='cyk')
            self.lex = self.instance.lex
            self.rules = self.parser.grammar.rules

        else:
            # self.parser = lark.parsers.earley.Parser(rules)
            self.instance = lark.Lark(grammar, parser='earley')
            self.lex = self.instance.lex
            self.rules = rules

        self.terminals = terminals
        self.ignores = ignores

    def transducer(self, decay=0.99, **kwargs):
        """
        XXX: Warning: There may be infelicity in the tokenization semantics as there is
        no longer a prioritized or maximum munch semantics to tokenizer.  It is
        probabilistic and the weights are set pretty arbitrarily.
        """
        from genparse import EPSILON, FST, Float

        m = FST(Float)
        START = 0
        STOP = 1
        m.add_I(START, 1)
        m.add_F(STOP, decay)
        m.add_arc(STOP, (EPSILON, EPSILON), START, 1)
        for token_id, token_class in enumerate(self.terminals):
            fsm = regex_to_greenery(token_class.pattern.to_regexp(), **kwargs)
            m.add_arc(START, (EPSILON, token_class.name), (token_id, fsm.initial), 1)
            for final_state in fsm.finals:
                m.add_arc((token_id, final_state), (EPSILON, EPSILON), STOP, 1)
            dead = {i for i in fsm.states if not fsm.islive(i)}
            for state in fsm.states:
                arcs = fsm.map[state]
                for input_char, next_state in arcs.items():
                    if next_state in dead:
                        continue
                    for char in input_char.get_chars():
                        m.add_arc(
                            (token_id, state),
                            (char, EPSILON),
                            (token_id, next_state),
                            decay,
                        )
        return m

    def convert(self):
        "Convert the lark grammar into a `genparse.CFG` grammar."
        from genparse import CFG, Float, Rule

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

    def char_cfg(self, decay, ignore=''):
        from genparse import CFG, Float

        cfg = self.convert()

        foo = CFG(Float, S=cfg.S, V=set())
        for r in cfg:
            foo.add(r.w, r.head, *r.body)

        for token_class in self.terminals:
            regex = token_class.pattern.to_regexp()
            if ignore:
                regex += ignore

            fsa = greenery_to_wfsa(
                regex, decay=decay, name=lambda x, t=token_class.name: (t, x)
            )
            # display(fsa)
            G = fsa.to_cfg(S=token_class.name)

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


def expand_case_insensitive(r):
    """
    Lark accepts case-insensitive terminals of the form `".*"i`
    In python re syntax, these compile to `(?i:.*)`
    This function desugars the latter into a format supported by greenery,
    Supporting arbitrary nesting of case insensitive contexts,
    And does so in a single O(len(r)) scan.
    """
    end = len(r)
    last3 = ('', '', '')
    state = 0
    count = 0
    depth = 0
    ptr = 0
    out = ''
    fix_sugar = any(_ in r for _ in ('[a-z]', '[A-Z]', '[a-zA-Z]'))
    while True:
        if ptr == end:
            if fix_sugar:
                out = out.replace('[[aA]-[zZ]]', '[a-zA-Z]').replace(
                    '[[aA]-[zZ][aA]-[zZ]]', '[a-zA-Z]'
                )
            return out
        c = r[ptr]
        if state == 0:
            if c == ':' and ''.join(last3) == '(?i':
                out = out[:-3]
                state = 1
                count = 1
            else:
                out += c
        elif state == 1:
            if c.isalpha():
                if last3[2] == '\\' and last3[1] != '\\':
                    out += c
                else:
                    out += f'[{c.lower()}{c.upper()}]'
            elif c == ':' and ''.join(last3) == '(?i':
                out = out[:-6]
                depth += 1
            elif c == ']':
                if ''.join(last3) == f'[{last3[1].lower()}{last3[1].upper()}':
                    out = out[:-8] + out[-7:-4]
                else:
                    out += c
            elif c == '(':
                count += 1
                out += c
            elif c == ')':
                count -= 1
                if count == 0:
                    state = 0
                elif count == depth:
                    depth -= 1
                else:
                    out += c
            else:
                out += c
        else:
            raise ValueError('invalid state')
        last3 = (last3[1], last3[2], c)
        ptr += 1


def regex_to_greenery(regex, ignore=''):
    """
    Convert `regex`, a python-like regular expression (`re`), into a `greenery`
    finite-state machine (FSM).
    """
    import greenery

    regex = expand_case_insensitive(regex)

    # Patch: note that greenery does not escape spaces but both the `re` and `lark` do.
    return greenery.parse(regex.replace('\\ ', ' ') + ignore).to_fsm()


# Not essential; only used in a notebook to visualize individual greenery FSMs
# def greenery_to_fsa(fsm):
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
def greenery_to_wfsa(fsm, decay=0.99, name=lambda x: x):
    from genparse import WFSA, Float

    if isinstance(fsm, str):
        fsm = regex_to_greenery(fsm)
    m = WFSA(Float)
    m.add_I(name(fsm.initial), 1)

    rejection_states = [e for e in fsm.states if not fsm.islive(e)]
    for state in fsm.states:
        arcs = fsm.map[state]

        # determine this state's fan out...
        K = 0
        for input_char, next_state in arcs.items():
            if next_state in rejection_states:
                continue  # rejection state
            for char in input_char.get_chars():
                K += 1
        if state in fsm.finals:
            K += 1

        if K == 0:
            continue

        if state in fsm.finals:
            m.add_F(name(state), decay / K)

        for input_char, next_state in arcs.items():
            if next_state in rejection_states:
                continue  # rejection state
            for char in input_char.get_chars():
                m.add_arc(name(state), char, name(next_state), decay / K)

    return m


def format_table(rows, headings=None):
    def fmt(x):
        if hasattr(x, '_repr_html_'):
            return x._repr_html_()
        elif hasattr(x, '_repr_svg_'):
            return x._repr_svg_()
        elif hasattr(x, '_repr_image_svg_xml'):
            return x._repr_image_svg_xml()
        else:
            return f'<pre>{html.escape(str(x))}</pre>'

    return (
        '<table>'
        + (
            '<tr style="font-weight: bold;">'
            + ''.join(f'<td>{x}</td>' for x in headings)
            + '</tr>'
            if headings
            else ''
        )
        + ''.join(
            '<tr>' + ''.join(f'<td>{fmt(x)}</td>' for x in row) + ' </tr>' for row in rows
        )
        + '</table>'
    )


def display_table(*args, **kwargs):
    return display(HTML(format_table(*args, **kwargs)))
