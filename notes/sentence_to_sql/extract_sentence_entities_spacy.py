"""
Extract entities from some input sentences.
"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Any

import spacy

logger = logging.getLogger(__name__)

PERSON_LABEL = 'PERSON'


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(mode='r', encoding='utf-8') as jsonl_in:
        result = [json.loads(line) for line in jsonl_in if line.strip()]
    return result


def save_jsonl(outputs: list[dict[str, Any]], path: Path) -> None:
    with path.open(mode='w', encoding='utf-8') as jsonl_out:
        for output in outputs:
            json.dump(output, jsonl_out)
            jsonl_out.write('\n')


def _uniquify(items):
    """
    O(n^2) order-preserving uniquification.

    Fine for short lists like a single sentence's list of PERSON entities.
    """
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('sentences_path', type=Path, help='Path to the prompts file.')
    parser.add_argument(
        'save_entities_to',
        type=Path,
        help='Where to save the JSONL file of sentences annotated with entities.',
    )
    parser.add_argument(
        '--spacy-model',
        default='en_core_web_sm',
        help='The spaCy model to use for NER.',
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
    save_entities_to: Path = args.save_entities_to
    spacy_model: str = args.spacy_model

    nlp = spacy.load(spacy_model)

    sentence_data = load_jsonl(sentences_path)
    spacy_parses = [nlp(sentence_datum['sentence']) for sentence_datum in sentence_data]
    outputs = [
        {
            **sentence_datum,
            'entities': {
                'people': _uniquify(
                    [ent.text for ent in doc.ents if ent.label_ == PERSON_LABEL]
                )
            },
            'entities_metadata': {
                'model': spacy_model,
            },
        }
        for sentence_datum, doc in zip(sentence_data, spacy_parses)
    ]
    save_jsonl(outputs, save_entities_to)


if __name__ == '__main__':
    main()
