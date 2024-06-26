"""
Converts sentences into prompts to generate SQL queries about those prompts.
"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
import string
from typing import Any

logger = logging.getLogger(__name__)


def sentence_prompt(sentence_dict: dict[str, Any]) -> dict[str, Any]:
    """Use the sentence itself as prompt with no instructions."""
    return {'prompt': sentence_dict['sentence']}


_INSTRUCTION_TEMPLATE_V1_PERSONONLY = string.Template(
    """Write a SQL query about a person or an organization mentioned in the following sentence:

$sentence

SQL query:"""
)


def instruction_prompt_v1_persononly(sentence_dict: dict[str, Any]) -> dict[str, Any]:
    """Use the sentence itself as prompt with no instructions."""
    return {
        'prompt': _INSTRUCTION_TEMPLATE_V1_PERSONONLY.substitute(
            sentence=sentence_dict['sentence']
        )
    }


_PROMPT_TEMPLATES = {
    'sentence': sentence_prompt,
    'instruction-v1-persononly': instruction_prompt_v1_persononly,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(mode='r', encoding='utf-8') as jsonl_in:
        result = [json.loads(line) for line in jsonl_in if line.strip()]
    return result


def save_jsonl(outputs: list[dict[str, Any]], path: Path) -> None:
    with path.open(mode='w', encoding='utf-8') as jsonl_out:
        for output in outputs:
            json.dump(output, jsonl_out)
            jsonl_out.write('\n')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        'sentences_path', type=Path, help='Path to the file of sentences.'
    )
    parser.add_argument(
        'save_prompts_to', type=Path, help='Where to save the output prompts.'
    )
    parser.add_argument(
        '--prompt-template',
        default='sentence',
        choices=_PROMPT_TEMPLATES,
        help='The prompt template to use.',
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

    sentences_path: Path = args.sentences_path
    save_prompts_to: Path = args.save_prompts_to
    prompt_template: str = args.prompt_template

    sentence_dicts = load_jsonl(sentences_path)
    promptify = _PROMPT_TEMPLATES[prompt_template]
    prompt_dicts = [
        {**sentence_dict, **promptify(sentence_dict)} for sentence_dict in sentence_dicts
    ]
    save_jsonl(prompt_dicts, save_prompts_to)


if __name__ == '__main__':
    main()
