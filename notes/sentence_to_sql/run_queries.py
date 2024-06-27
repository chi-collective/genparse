"""
Runs SQL queries generated using Genparse against a database.
"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)


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
    parser.add_argument('inferences_path', type=Path, help='Path to the inferences file.')
    parser.add_argument('database_path', type=Path, help='Path to the database file.')
    parser.add_argument(
        'save_query_results_to',
        type=Path,
        help='Where to save the JSONL file of query results.',
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

    inferences_path: Path = args.inferences_path
    database_path: Path = args.database_path
    save_query_results_to: Path = args.save_query_results_to

    inference_dicts = load_jsonl(inferences_path)
    connection = sqlite3.connect(database_path)
    with connection.cursor() as cursor:
        outputs = [
            {
                **inference_dict,
                'inferred_query_results': {
                    query: cursor.execute(query).fetchall()
                    for query in inference_dict['genparse_inference']
                },
            }
            for inference_dict in inference_dicts
        ]
    save_jsonl(outputs, save_query_results_to)


if __name__ == '__main__':
    main()
