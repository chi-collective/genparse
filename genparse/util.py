import html
import numpy as np
import random
import torch
import transformers
import hfppl
import arsenal
from collections import Counter
from IPython.display import HTML, display

import string
import interegular
from interegular.fsm import anything_else


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def top_p_filter(p, top_p):
    """
    Implemented top-p filtering, aka Nucleus Sampling.

    >>> P = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1-(1/2 + 1/4 + 1/8 + 1/16 + 1/32 + 1/64)]

    >>> top_p_filter(P, 1.0)
    array([0.5     , 0.25    , 0.125   , 0.0625  , 0.03125 , 0.015625,
           0.015625])

    >>> top_p_filter(P, 0.75)
    array([0.66666667, 0.33333333, 0.        , 0.        , 0.        ,
           0.        , 0.        ])

    >>> top_p_filter(P, 0.5)
    array([1., 0., 0., 0., 0., 0., 0.])

    >>> top_p_filter(P, 0.0001)
    array([1., 0., 0., 0., 0., 0., 0.])

    """
    assert 0 <= top_p <= 1

    p = np.asarray(p)
    assert np.allclose(np.sum(p), 1), np.sum(p)

    order = np.argsort(p)[::-1]

    total = 0
    for i in order:
        if total >= top_p:
            p[i] = 0
        total += p[i]

    Z = p.sum()

    p /= Z

    return p


def lark_guide(grammar, **kwargs):
    from genparse import BoolCFGLM

    return BoolCFGLM(LarkStuff(grammar).char_cfg(**kwargs))


def load_model_by_name(model_name, batch_size=None, temperature=1, top_p=None):
    """
    Load an LLM from 🤗 into a genparse `TokenizedLLM`.

    Adding "mock-" will create an imitation model over the same vocabulary
    that can be used for testing.
    """
    from genparse.lm import TokenizedLLM, LLM, MockLLM
    from genparse.tokenization import decode_tokenizer_vocab

    if model_name == 'gpt2':
        MODEL_ID = 'gpt2'
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID)
        return TokenizedLLM(
            model=LLM(
                model, V=set(range(tokenizer.vocab_size)), eos=tokenizer.eos_token_id
            ),
            tokenizer=tokenizer,
            batch_size=batch_size,
            temperature=temperature,
            top_p=top_p,
        )

    elif model_name == 'mock-gpt2':
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
        return TokenizedLLM(
            model=MockLLM(
                V=decode_tokenizer_vocab(tokenizer),
                eos=tokenizer.eos_token,
            ),
            tokenizer=tokenizer,
            batch_size=batch_size,
            temperature=temperature,
            top_p=top_p,
        )

    elif model_name == 'mock-codellama':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            'codellama/CodeLlama-7b-Instruct-hf'
        )
        return TokenizedLLM(
            model=MockLLM(
                V=decode_tokenizer_vocab(tokenizer),
                eos=tokenizer.eos_token,
            ),
            tokenizer=tokenizer,
            batch_size=batch_size,
            temperature=temperature,
            top_p=top_p,
        )

    elif model_name == 'codellama':
        MODEL_ID = 'codellama/CodeLlama-7b-Instruct-hf'
        return TokenizedLLM(
            model=hfppl.CachedCausalLM.from_pretrained(MODEL_ID, load_in_8bit=False),
            tokenizer=transformers.AutoTokenizer.from_pretrained(
                MODEL_ID,
                use_fast=True,
                eot_token=None,
                fill_token=None,
                prefix_token=None,
                middle_token=None,
                suffix_token=None,
            ),
            batch_size=batch_size,
            temperature=temperature,
            top_p=top_p,
        )

    else:
        raise ValueError(model_name)


class InferenceSetup:
    def __init__(
        self,
        model_name,
        grammar,
        proposal_name='character',
        seed=None,
        batch_size=None,
        guide_opts=None,
        proposal_opts=None,
        temperature=1,
    ):
        from genparse.steer import HFPPLSampler
        from genparse.proposal import CharacterProposal, TokenProposal

        if guide_opts is None:
            guide_opts = {}
        if proposal_opts is None:
            proposal_opts = {}

        if seed is not None:
            set_seed(seed)

        llm = load_model_by_name(
            model_name, batch_size=batch_size, temperature=temperature
        )
        guide = lark_guide(grammar, **guide_opts)
        sampler = HFPPLSampler(llm=llm, guide=guide)

        if proposal_name == 'character':
            proposal = CharacterProposal(llm=llm, guide=guide, **proposal_opts)
        elif proposal_name == 'token':
            proposal = TokenProposal(llm=llm, guide=guide, **proposal_opts)
        else:
            raise ValueError(f'invalid proposal name {proposal!r}')

        self.sampler = sampler
        self.proposal = proposal

    def __call__(
        self, prompt, n_particles, method='smc-standard', max_tokens=1000, **kwargs
    ):
        return self.sampler.run_inference(
            prompt=prompt,
            proposal=self.proposal,
            method=method,
            n_particles=n_particles,
            max_tokens=max_tokens,
            **kwargs,
        )


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
        import lark

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

    def char_cfg(self, decay=None, delimiter='', charset='core'):
        from genparse import CFG, Float

        if decay is not None:
            import warnings

            warnings.warn('Option `decay` is deprecated')

        if delimiter:
            import warnings

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


# def expand_case_insensitive(r):
#    """
#    Lark accepts case-insensitive terminals of the form `".*"i`
#    In python re syntax, these compile to `(?i:.*)`
#    This function desugars the latter into a format supported by greenery,
#    Supporting arbitrary nesting of case insensitive contexts,
#    And does so in a single O(len(r)) scan.
#    """
#    end = len(r)
#    last3 = ('', '', '')
#    state = 0
#    count = 0
#    depth = 0
#    ptr = 0
#    out = ''
#    fix_sugar = any(_ in r for _ in ('[a-z]', '[A-Z]', '[a-zA-Z]'))
#    while True:
#        if ptr == end:
#            if fix_sugar:
#                out = out.replace('[[aA]-[zZ]]', '[a-zA-Z]').replace(
#                    '[[aA]-[zZ][aA]-[zZ]]', '[a-zA-Z]'
#                )
#            return out
#        c = r[ptr]
#        if state == 0:
#            if c == ':' and ''.join(last3) == '(?i':
#                out = out[:-3]
#                state = 1
#                count = 1
#            else:
#                out += c
#        elif state == 1:
#            if c.isalpha():
#                if last3[2] == '\\' and last3[1] != '\\':
#                    out += c
#                else:
#                    out += f'[{c.lower()}{c.upper()}]'
#            elif c == ':' and ''.join(last3) == '(?i':
#                out = out[:-6]
#                depth += 1
#            elif c == ']':
#                if ''.join(last3) == f'[{last3[1].lower()}{last3[1].upper()}':
#                    out = out[:-8] + out[-7:-4]
#                else:
#                    out += c
#            elif c == '(':
#                count += 1
#                out += c
#            elif c == ')':
#                count -= 1
#                if count == 0:
#                    state = 0
#                elif count == depth:
#                    depth -= 1
#                else:
#                    out += c
#            else:
#                out += c
#        else:
#            raise ValueError('invalid state')
#        last3 = (last3[1], last3[2], c)
#        ptr += 1


# def regex_to_greenery(regex):
#    """
#    Convert `regex`, a python-like regular expression (`re`), into a `greenery`
#    finite-state machine (FSM).
#    """
#    import greenery
#
#    regex = expand_case_insensitive(regex)
#
#    # Patch: note that greenery does not escape spaces but both the `re` and `lark` do.
#    return greenery.parse(regex.replace('\\ ', ' ')).to_fsm()


# def greenery_to_wfsa(fsm, decay=1, name=lambda x: x, charset='core'):
#    from genparse import WFSA, Float
#
#    if charset == 'core':
#        import string
#
#        charset = set(string.printable)
#    else:
#        # TODO: implement other charsets
#        raise NotImplementedError(f'charset {charset} not implemented')
#
#    if isinstance(fsm, str):
#        fsm = regex_to_greenery(fsm)
#    m = WFSA(Float)
#    m.add_I(name(fsm.initial), 1)
#
#    rejection_states = [e for e in fsm.states if not fsm.islive(e)]
#    for state in fsm.states:
#        arcs = fsm.map[state]
#
#        # determine this state's fan out...
#        K = 0
#        for input_char, next_state in arcs.items():
#            if next_state in rejection_states:
#                continue  # rejection state
#            if input_char.negated:
#                chars = charset - set(input_char.get_chars())
#            else:
#                chars = input_char.get_chars()
#            for char in chars:
#                K += 1
#        if state in fsm.finals:
#            K += 1
#
#        if K == 0:
#            continue
#
#        if state in fsm.finals:
#            m.add_F(name(state), decay / K)
#
#        for input_char, next_state in arcs.items():
#            if next_state in rejection_states:
#                continue  # rejection state
#            if input_char.negated:
#                chars = charset - set(input_char.get_chars())
#            else:
#                chars = input_char.get_chars()
#            for char in chars:
#                m.add_arc(name(state), char, name(next_state), decay / K)
#
#    return m


def interegular_to_wfsa(pattern, name=lambda x: x, charset='core'):
    from genparse import WFSA, Float

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
                assert isinstance(A, str) and len(A) == 1
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
