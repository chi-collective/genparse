import html
import numpy as np
import random
import torch
import transformers
import hfppl
from IPython.display import HTML, display


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
    from genparse.lark_interface import LarkStuff

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
        llm_opts=None,
    ):
        from genparse.steer import HFPPLSampler
        from genparse.proposal import CharacterProposal, TokenProposal

        if guide_opts is None:
            guide_opts = {}
        if proposal_opts is None:
            proposal_opts = {}
        if llm_opts is None:
            llm_opts = {}

        if seed is not None:
            set_seed(seed)

        llm = load_model_by_name(model_name, batch_size=batch_size, **llm_opts)
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
