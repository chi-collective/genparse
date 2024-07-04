import transformers
import torch
from genparse.util import set_seed, lark_guide
from genparse.proposal import CharacterProposal, TokenProposal

from genparse.backends.vllm.llm import vllmpplLLM
from genparse.backends.vllm.steer import VLLMSampler


class InferenceSetupVLLM:
    def __init__(
        self,
        model_name,
        grammar,
        proposal_name='character',
        seed=None,
        guide_opts=None,
        proposal_opts=None,
        batch_size=None,
    ):
        from genparse.lm import TokenizedLLM

        if guide_opts is None:
            guide_opts = {}
        if proposal_opts is None:
            proposal_opts = {}

        if seed is not None:
            set_seed(seed)

        torch.backends.cuda.matmul.allow_tf32 = True

        if model_name == 'gpt2':
            MODEL_ID = 'gpt2'
            llm = TokenizedLLM(
                model=vllmpplLLM(MODEL_ID),
                tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL_ID),
                batch_size=batch_size,
            )

        elif model_name == 'codellama':
            MODEL_ID = 'codellama/CodeLlama-7b-Instruct-hf'
            llm = TokenizedLLM(
                model=vllmpplLLM(MODEL_ID, dtype=torch.float32, max_model_len=4096),
                tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL_ID),
                batch_size=batch_size,
            )

        else:
            raise ValueError(model_name)

        guide = lark_guide(grammar, **guide_opts)
        sampler = VLLMSampler(llm=llm, guide=guide)

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
