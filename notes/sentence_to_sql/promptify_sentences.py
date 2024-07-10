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
_INSTRUCTION_TEMPLATE_V2_PERSONONLY = string.Template(
    """Write a SQL query about a famous person mentioned in the following sentence:

$sentence

The database has one table, People, with columns as follows:
 
- "itemLabel.value" (usually the person's common name, which may or may not match given + family name, e.g. Ada Lovelace vs. Ada King)
- "given_nameLabel.value" (given name)
- "family_nameLabel.value" (family name)
- "name_in_native_languageLabel.value" (name as written in native language)
- "languages_spoken__written_or_signedLabel.value" (language they speak, write, or sign)
- "date_of_birthLabel.value" (their date of birth, in YYYY-MM-DDTHH:mm:ss+XX:ZZ form)
- "place_of_birthLabel.value" (their place of birth)
- "spouseLabel.value" (their spouse's common name)
- "motherLabel.value" (their mother's common name)
- "fatherLabel.value" (their father's common name)
- "country_of_citizenshipLabel.value" (country they are a citizen of)
- "occupationLabel.value" (their occupation)
- "religion_or_worldviewLabel.value" (their religion or worldview)
- "sex_or_genderLabel.value" (their sex or gender)

The dots are part of the column names and so the column names must be quoted.

Note that a person may appear in multiple rows if, for example, they have more than one occupation, speak more than one language, or are citizens of multiple countries.

SQL query:"""
)

_INSTRUCTION_TEMPLATE_V3_PERSONONLY_WITH_ENTITIES = string.Template(
    """Write a SQL query about a famous person mentioned in the following sentence:

$sentence

The database has one table, People, with columns as follows:

- "itemLabel.value" (usually the person's common name, which may or may not match given + family name, e.g. Ada Lovelace vs. Ada King)
- "given_nameLabel.value" (given name)
- "family_nameLabel.value" (family name)
- "name_in_native_languageLabel.value" (name as written in native language)
- "languages_spoken__written_or_signedLabel.value" (language they speak, write, or sign)
- "date_of_birthLabel.value" (their date of birth, in YYYY-MM-DDTHH:mm:ss+XX:ZZ form)
- "place_of_birthLabel.value" (their place of birth)
- "spouseLabel.value" (their spouse's common name)
- "motherLabel.value" (their mother's common name)
- "fatherLabel.value" (their father's common name)
- "country_of_citizenshipLabel.value" (country they are a citizen of)
- "occupationLabel.value" (their occupation)
- "religion_or_worldviewLabel.value" (their religion or worldview)
- "sex_or_genderLabel.value" (their sex or gender)

The dots are part of the column names and so the column names must be quoted.

Note that a person may appear in multiple rows if, for example, they have more than one occupation, speak more than one language, or are citizens of multiple countries.

The people named in the sentence are:

$people

SQL query:"""
)

_INSTRUCTION_TEMPLATE_V3_PERSONONLY_WITH_ENTITIES_ONLY = string.Template(
    """Write a SQL query about one of the following people:

$people

The database has one table, People, with columns as follows:

- "itemLabel.value" (usually the person's common name, which may or may not match given + family name, e.g. Ada Lovelace vs. Ada King)
- "given_nameLabel.value" (given name)
- "family_nameLabel.value" (family name)
- "name_in_native_languageLabel.value" (name as written in native language)
- "languages_spoken__written_or_signedLabel.value" (language they speak, write, or sign)
- "date_of_birthLabel.value" (their date of birth, in YYYY-MM-DDTHH:mm:ss+XX:ZZ form)
- "place_of_birthLabel.value" (their place of birth)
- "spouseLabel.value" (their spouse's common name)
- "motherLabel.value" (their mother's common name)
- "fatherLabel.value" (their father's common name)
- "country_of_citizenshipLabel.value" (country they are a citizen of)
- "occupationLabel.value" (their occupation)
- "religion_or_worldviewLabel.value" (their religion or worldview)
- "sex_or_genderLabel.value" (their sex or gender)

The dots are part of the column names and so the column names must be quoted.

Note that a person may appear in multiple rows if, for example, they have more than one occupation, speak more than one language, or are citizens of multiple countries.

SQL query:"""
)

_INSTRUCTION_TEMPLATE_V4_PERSONONLY_WITH_ENTITIES = string.Template(
    """Write a SQL query about a famous person mentioned in the following sentence:

$sentence

The database has one table, People, with columns as follows:

- "given_nameLabel.value" (given name)
- "family_nameLabel.value" (family name)
- "name_in_native_languageLabel.value" (name as written in native language)
- "languages_spoken__written_or_signedLabel.value" (language they speak, write, or sign)
- "date_of_birthLabel.value" (their date of birth, in YYYY-MM-DDTHH:mm:ss+XX:ZZ form)
- "place_of_birthLabel.value" (their place of birth)
- "spouseLabel.value" (their spouse's common name)
- "motherLabel.value" (their mother's common name)
- "fatherLabel.value" (their father's common name)
- "country_of_citizenshipLabel.value" (country they are a citizen of)
- "occupationLabel.value" (their occupation)
- "religion_or_worldviewLabel.value" (their religion or worldview)
- "sex_or_genderLabel.value" (their sex or gender)

The dots are part of the column names and so the column names must be quoted.

Note that a person may appear in multiple rows if, for example, they have more than one occupation, speak more than one language, or are citizens of multiple countries.

The people named in the sentence are:

$people

SQL query:"""
)


def instruction_prompt_v1_persononly(sentence_dict: dict[str, Any]) -> dict[str, Any]:
    """Use the sentence itself as prompt with no instructions."""
    return {
        'prompt': _INSTRUCTION_TEMPLATE_V1_PERSONONLY.substitute(
            sentence=sentence_dict['sentence']
        )
    }


def instruction_prompt_v2_persononly(sentence_dict: dict[str, Any]) -> dict[str, Any]:
    """Provide the sentence, basic instructions, and a list of columns that can be queried."""
    return {
        'prompt': _INSTRUCTION_TEMPLATE_V2_PERSONONLY.substitute(
            sentence=sentence_dict['sentence']
        )
    }


def _format_as_list(items: list[str]) -> str:
    """Format strings into a Markdown list (with no trailing newline)."""
    with_bullets = [f'- {item}' for item in items]
    return '\n'.join(with_bullets)


def instruction_prompt_v3_persononly_with_entities(
    sentence_dict: dict[str, Any],
) -> dict[str, Any]:
    """Provide the sentence, the names, basic instructions, and a list of columns that can be queried."""
    return {
        'prompt': _INSTRUCTION_TEMPLATE_V3_PERSONONLY_WITH_ENTITIES.substitute(
            sentence=sentence_dict['sentence'],
            people=_format_as_list(sentence_dict['entities']['people']),
        )
    }


def instruction_prompt_v3_persononly_with_entities_only(
    sentence_dict: dict[str, Any],
) -> dict[str, Any]:
    """Provide the names only (not the sentence), the instructions, and a list of columns that can be queried."""
    return {
        'prompt': _INSTRUCTION_TEMPLATE_V3_PERSONONLY_WITH_ENTITIES_ONLY.substitute(
            people=_format_as_list(sentence_dict['entities']['people'])
        )
    }


def instruction_prompt_v4_persononly_with_entities(
    sentence_dict: dict[str, Any],
) -> dict[str, Any]:
    """Provide the sentence, the names, basic instructions, and a list of columns that can be queried."""
    return {
        'prompt': _INSTRUCTION_TEMPLATE_V4_PERSONONLY_WITH_ENTITIES.substitute(
            sentence=sentence_dict['sentence'],
            people=_format_as_list(sentence_dict['entities']['people']),
        )
    }


_PROMPT_TEMPLATES = {
    'sentence': sentence_prompt,
    'instruction-v1-persononly': instruction_prompt_v1_persononly,
    'instruction-v2-persononly': instruction_prompt_v2_persononly,
    'instruction-v3-persononly-with-entities': instruction_prompt_v3_persononly_with_entities,
    'instruction-v3-persononly-with-entities-only': instruction_prompt_v3_persononly_with_entities_only,
    'instruction-v4-persononly-with-entities': instruction_prompt_v4_persononly_with_entities,
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
