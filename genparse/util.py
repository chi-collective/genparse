import os
import html
import torch
import random
import logging
import warnings
import numpy as np
import transformers
import multiprocessing as mp
from IPython.display import HTML, display


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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


def lark_guide_fast(grammar, **kwargs):
    from genparse.experimental.earley_fast import BoolCFGLM
    from genparse.lark_interface import LarkStuff

    return BoolCFGLM(LarkStuff(grammar).char_cfg(**kwargs))


def make_model_table(name2id):
    name_width = max(len('Name'), max(len(name) for name in name2id.keys()))
    id_width = max(len('HuggingFace Identifier'), max(len(id) for id in name2id.values()))
    table = f"{'Name':<{name_width}} | {'HuggingFace Identifier':<{id_width}}\n"
    table += f"{'-' * name_width}-+-{'-' * id_width}\n"
    for name, id in name2id.items():
        table += f'{name:<{name_width}} | {id:<{id_width}}\n'
    return table


def load_model_by_name(model_name, use_vllm=False, vllm_engine_opts={}, **kwargs):
    """
    Load an LLM from 🤗 into a genparse LLM.

    Adding a "mock-" prefix to a name will create an imitation model over the same vocabulary
    that can be used for testing.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from genparse.lm import VirtualTokenizedLLM, TokenizedLLM, LLM, MockLLM
    from genparse.tokenization import decode_tokenizer_vocab

    name2id = {
        'llama3': 'meta-llama/Meta-Llama-3-8B',
        'llama3.1': 'meta-llama/Meta-Llama-3.1-8B',
        'llama3-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'llama3.1-instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'codellama': 'codellama/CodeLlama-7b-Instruct-hf',
        'gpt2': 'gpt2',
        'gpt2-medium': 'gpt2-medium',
        'gpt2-large': 'gpt2-large',
    }

    is_mock = model_name.startswith('mock-')

    if is_mock:
        model_name = model_name.split('mock-')[1]
        use_vllm = False

    if model_name not in name2id:
        raise ValueError(
            f'Invalid model name {model_name!r}. '
            f'GenParse supports the following models:\n\n{make_model_table(name2id)}\n'
            'Adding a "mock-" prefix to a name will create an imitation model over the same vocabulary '
            'that can be used for testing.'
        )

    model_id = name2id[model_name]

    if is_mock:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return TokenizedLLM(
            model=MockLLM(
                V=decode_tokenizer_vocab(tokenizer),
                eos=tokenizer.eos_token,
            ),
            tokenizer=tokenizer,
            **kwargs,
        )
    elif use_vllm:
        return VirtualTokenizedLLM.from_name(model_id, vllm_engine_opts, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return TokenizedLLM(
            model=LLM(
                AutoModelForCausalLM.from_pretrained(model_id),
                V=decode_tokenizer_vocab(tokenizer),
                eos=tokenizer.eos_token_id,
            ),
            tokenizer=tokenizer,
            **kwargs,
        )


class InferenceSetup:
    def __init__(
        self,
        model_name,
        grammar,
        proposal_name='character',
        num_processes=min(mp.cpu_count(), 2),
        use_rust_parser=True,
        use_vllm=None,
        seed=None,
        guide_opts=None,
        proposal_opts=None,
        llm_opts=None,
        vllm_engine_opts=None,
    ):
        """
        Initialize an InferenceSetup object for running inference with a specified model and grammar.

        Args:
            model_name (str): Name of the language model to use.
            grammar (str): The grammar specification in Lark format.
            proposal_name (str, optional): Type of proposal to use ('character' or 'token'). Defaults to 'character'.
            num_processes (int, optional): Number of processes to use for parallel proposals.
                Defaults to min(CPU count, 2).
            use_rust_parser (bool, optional): Whether to use Rust implementation of Earley parser for faster inference.
                Defaults to True. When False, Python implementation is used.
            use_vllm (bool, optional): Whether to use VLLM for LLM next token probability computations.
                Defaults to None, which uses VLLM when possible.
            seed (int, optional): Random seed for reproducibility.
            guide_opts (dict, optional): Additional options for the guide.
            proposal_opts (dict, optional): Additional options for the proposal (e.g., K for token proposal).
            llm_opts (dict, optional): Additional options for the genparse LM (e.g., temperature, top_p).
            vllm_engine_opts (dict, optional): Additional options for the VLLM engine (e.g., dtype).
                These are ignored when VLLM is not used.

        Example usage:
            # Create an InferenceSetup object for Llama 3.1 with a simple grammar and the character proposal
            model = InferenceSetup(
                model_name="llama3.1",
                grammar='start: "Sequential Monte Carlo is " ( "good" | "bad" )',
                proposal_name="character",
            )
            particle_approx = model('Say something nice about SMC:', n_particles=10, verbosity=1)

            # Update grammar
            model.update_grammar('start: "Sequential Monte Carlo is " ( "pretty good" | "great" )')
            particle_approx = model('Say something nice about SMC:', n_particles=10, verbosity=1)

        """
        from genparse.lm import VirtualTokenizedLLM

        self.model_name = model_name
        self.grammar = grammar
        self.num_processes = num_processes
        self.proposal_name = proposal_name
        self.use_vllm = use_vllm
        self.use_rust_parser = use_rust_parser
        self.seed = seed
        self.guide_opts = guide_opts or {}
        self.proposal_opts = proposal_opts or {}
        self.llm_opts = llm_opts or {}
        self.vllm_engine_opts = vllm_engine_opts or {}

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s : %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        proposal_names = ['character', 'token']

        if proposal_name not in proposal_names:
            raise ValueError(
                f"Invalid proposal name {proposal_name!r}. "
                f"proposal_name must be one of `{' `'.join(proposal_names)}`"
            )

        if seed is not None:
            set_seed(seed)

        if self.use_vllm is None:
            try:  # check whether we can use vllm
                import vllm

                self.use_vllm = torch.cuda.is_available()
            except ImportError:
                self.use_vllm = False

        self.llm = load_model_by_name(
            self.model_name,
            use_vllm=self.use_vllm,
            vllm_engine_opts=self.vllm_engine_opts,
            **self.llm_opts,
        )

        if isinstance(self.llm, VirtualTokenizedLLM):
            from genparse.batch_inference.lm import BatchVLLM

            self.logger.info('Using VLLM for LM next token probability computations')
            self.batch_llm = BatchVLLM(self.llm)
        else:
            from genparse.batch_inference.lm import BatchLLM

            self.logger.info('Using CPU for LM next token probability computations')
            self.batch_llm = BatchLLM(self.llm)

        self.batch_model = self._init_step_model()

    def _init_step_model(self):
        from genparse.batch_inference.steer import BatchStepModel

        if self.use_rust_parser:
            self.logger.info('Initializing Rust Earley parser')
            self.guide = lark_guide_fast(self.grammar, **self.guide_opts)
        else:
            self.logger.info('Initializing Python Earley parser')
            self.guide = lark_guide(self.grammar, **self.guide_opts)

        self.logger.info(
            f'Initializing {self.proposal_name} proposal with {self.num_processes} subprocesses'
        )
        if self.num_processes > 1:
            if self.proposal_name == 'character':
                from genparse.batch_inference.proposal import ParallelCharacterProposal

                proposal = ParallelCharacterProposal
            elif self.proposal_name == 'token':
                from genparse.batch_inference.proposal import ParallelTokenProposal

                proposal = ParallelTokenProposal
            else:
                raise ValueError(self.proposal_name)

            self.batch_proposal = proposal(
                llm=self.llm,
                guide=self.guide,
                num_processes=self.num_processes,
                seed=self.seed,
                max_n_particles=500,
                **self.proposal_opts,
            )
        else:
            if self.proposal_name == 'character':
                from genparse.batch_inference.proposal import CharacterBatchProposal

                proposal = CharacterBatchProposal
            elif self.proposal_name == 'token':
                from genparse.batch_inference.proposal import TokenBatchProposal

                proposal = TokenBatchProposal
            else:
                raise ValueError(self.proposal_name)

            self.batch_proposal = proposal(
                llm=self.llm, guide=self.guide, **self.proposal_opts
            )

        return BatchStepModel(
            batch_proposal=self.batch_proposal,
            batch_llm=self.batch_llm,
            max_tokens=np.inf,
        )

    def __call__(self, prompt, n_particles, method='smc', max_tokens=500, **kwargs):
        """
        Run inference with n_particles using the specified method.

        Args:
            prompt (str): The input prompt to generate samples from.
            n_particles (int): The number of particles (samples) to generate.
            method (str, optional): The sampling method to use. Either 'smc' for sequential Monte Carlo
                                    or 'is' for importance sampling. Defaults to 'smc'.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 500.
            **kwargs: Additional keyword arguments to pass to the inference method.

        Returns:
            A ParticleApproximation.
        """
        from genparse.batch_inference.steer import smc, importance_sampling

        self.batch_model.max_tokens = max_tokens
        self.batch_model.set_prompt(prompt)

        if method == 'smc':
            sampler = smc
        elif method == 'is':
            sampler = importance_sampling
        else:
            raise ValueError(f'Invalid inference method {method!r}.')

        return sampler(batch_model=self.batch_model, n_particles=n_particles, **kwargs)

    def update_grammar(self, grammar):
        """
        Update the grammar used by the model.

        This method updates the grammar and reinitializes the batch model.

        Args:
            grammar: The new grammar to be used.

        Note:
            This method will clean up the resources (processes, shared memory)
            used by the current batch proposal.
        """
        self.grammar = grammar
        # cleanup shared resources and kill subprocesses
        self.batch_proposal.cleanup()
        self.batch_model = self._init_step_model()

    def cleanup(self):
        self.batch_model.cleanup()


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
