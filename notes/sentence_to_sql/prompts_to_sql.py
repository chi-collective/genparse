"""
Converts prompts into SQL queries about the people/orgs in those prompts.
"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Any

import transformers
import vllm

from genparse.cfglm import BoolMaskCFGLM
from genparse.lm import AsyncGreedilyTokenizedLLM
from genparse.proposal import CharacterProposal
from genparse.vllm_steer import VLLMSampler
from genparse.util import LarkStuff

logger = logging.getLogger(__name__)

# TODO load https://github.com/probcomp/genparse/blob/main/benchmark/grammars/sql_case_insensitive.lark
_SQL_GRAMMAR = """
start: "SELECT * FROM " location ";"

location: people_location | orgs_location
people_location: "People WHERE " people_clause ( " AND " people_clause ) *
orgs_location: "Organizations WHERE " orgs_clause ( " AND " orgs_clause ) *

people_clause: people_column " = " value
people_column: "first_name" | "last_name"
orgs_clause: orgs_column " = " value
orgs_column: ORG_NAME
ORG_NAME: "organization_name"

value: STRING
STRING: "\\"" CHAR * "\\""
CHAR: /(\\")|[^"]/

%import common.ESCAPED_STRING
"""


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
    model_name: str, prompts: list[str], *, batch_size: int, n_particles: int
) -> list[dict[str, float]]:
    llm = AsyncGreedilyTokenizedLLM(
        model=vllm.LLM(transformers.AutoModelForCausalLM.from_pretrained(model_name)),
        tokenizer=transformers.AutoTokenizer.from_pretrained(model_name),
        batch_size=batch_size,
    )
    guide = BoolMaskCFGLM(LarkStuff(_SQL_GRAMMAR).char_cfg(0.99, ignore='[ ]?'))
    proposal = CharacterProposal(llm=llm, guide=guide)
    sampler = VLLMSampler(llm=llm, guide=guide)
    result = [
        sampler.run_inference(
            prompt=prompt,
            proposal=proposal,
            method='smc-standard',
            n_particles=n_particles,
            verbosity=0,
        ).posterior
        for prompt in prompts
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
    n_particles: int = args.n_particles

    prompt_dicts = load_jsonl(prompts_path)
    inferences = run_inference(
        model,
        [prompt_dict['prompt'] for prompt_dict in prompt_dicts],
        batch_size=batch_size,
        n_particles=n_particles,
    )
    outputs = [
        {
            **prompt_dict,
            'genparse_inference': inference_dict,
            'inference_metadata': {
                'model': model,
                'batch_size': batch_size,
                'n_particles': n_particles,
            },
        }
        for prompt_dict, inference_dict in zip(prompt_dicts, inferences)
    ]
    save_jsonl(outputs, save_inferences_to)


if __name__ == '__main__':
    main()
