"""
Converts prompts into SQL queries about the people/orgs in those prompts.
"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Any

import transformers

from genparse.cfglm import BoolCFGLM
from genparse.lm import AsyncGreedilyTokenizedLLM
from genparse.proposal import CharacterProposal
from genparse.vllm_compatibility import vllmpplLLM
from genparse.vllm_steer import VLLMSampler
from genparse.util import LarkStuff

logger = logging.getLogger(__name__)
repo_root = Path(__file__).resolve().parent.parent.parent

_SQL_GRAMMAR = (Path(__file__).resolve().parent / 'ian_wikidata_sql.lark').read_text(
    encoding='utf-8'
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(mode='r', encoding='utf-8') as jsonl_in:
        result = [json.loads(line) for line in jsonl_in if line.strip()]
    return result


def save_jsonl(outputs: list[dict[str, Any]], path: Path) -> None:
    with path.open(mode='w', encoding='utf-8') as jsonl_out:
        for output in outputs:
            json.dump(output, jsonl_out)
            jsonl_out.write('\n')


def run_inference(
    model_name: str,
    prompts: list[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    n_particles: int,
) -> list[dict[str, float]]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = AsyncGreedilyTokenizedLLM(
        model=vllmpplLLM(model_name),
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    guide = BoolCFGLM(LarkStuff(_SQL_GRAMMAR).char_cfg(0.99, ignore='[ ]?'))
    proposal = CharacterProposal(llm=llm, guide=guide)
    sampler = VLLMSampler(llm=llm, guide=guide)
    reformatted_prompts = (
        [
            tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=False
            )
            for prompt in prompts
        ]
        if tokenizer.chat_template
        else prompts
    )
    result = [
        sampler.run_inference(
            prompt=prompt,
            proposal=proposal,
            method='smc-standard',
            max_tokens=max_new_tokens,
            n_particles=n_particles,
            verbosity=0,
        ).posterior
        for prompt in reformatted_prompts
    ]
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('prompts_path', type=Path, help='Path to the prompts file.')
    parser.add_argument(
        'save_inferences_to',
        type=Path,
        help='Where to save the JSONL file of inferences.',
    )
    parser.add_argument(
        '--model', default='gpt2', help='The language model to use for inference.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10, help='The batch size to use for sampling.'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=128,
        help='The maximum number of tokens to generate',
    )
    parser.add_argument(
        '--n-particles',
        type=int,
        default=15,
        help='The number of particles to use to approximate the posterior.',
    )
    parser.add_argument(
        '--logging-level',
        type=str,
        default='INFO',
        help='Logging level to use.',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.logging_level),
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    prompts_path: Path = args.prompts_path
    save_inferences_to: Path = args.save_inferences_to
    model: str = args.model
    batch_size: int = args.batch_size
    max_new_tokens: int = args.max_new_tokens
    n_particles: int = args.n_particles

    prompt_dicts = load_jsonl(prompts_path)
    inferences = run_inference(
        model,
        [prompt_dict['prompt'] for prompt_dict in prompt_dicts],
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        n_particles=n_particles,
    )
    outputs = [
        {
            **prompt_dict,
            'genparse_inference': inference_dict,
            'inference_metadata': {
                'model': model,
                'max_tokens': max_new_tokens,
                'batch_size': batch_size,
                'n_particles': n_particles,
            },
        }
        for prompt_dict, inference_dict in zip(prompt_dicts, inferences)
    ]
    save_jsonl(outputs, save_inferences_to)


if __name__ == '__main__':
    main()
