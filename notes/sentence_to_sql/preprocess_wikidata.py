"""
Preprocess Wikidata dumps into a nicer form for Code Llama's use.
"""

from argparse import ArgumentParser
import csv
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


DROP_COLUMNS = (
    '',
    'Unnamed: 0',
    'imageLabel.value',
    'instance_ofLabel.value',
    'Commons_categoryLabel.value',
    'ISNILabel.value',
    'signatureLabel.value',
    'described_by_sourceLabel.value',
    'topic_s_main_categoryLabel.value',
    'Commons_galleryLabel.value',
)
# These suffixes on a column name indicate it's a "data type" or "language" attribute.
# We don't care about those.
TYPE_TAG_SUFFIX = '.type'
XML_LANG_SUFFIX = '.xml:lang'
RENAME_COLUMNS = {
    'item': 'item_label',
}


def choose_columns_to_drop(fieldnames: Iterable[str]) -> set[str]:
    result = set(DROP_COLUMNS)
    result.update(
        column
        for column in fieldnames
        if column.endswith(TYPE_TAG_SUFFIX) or column.endswith(XML_LANG_SUFFIX)
    )
    return result


def clean_column(column: str) -> str:
    """Clean the given column name."""
    result = column.removesuffix('Label.value')
    result = RENAME_COLUMNS.get(result, result)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        'csv_in',
        type=Path,
        help='Path to the CSV Wikidata dump to preprocess.',
    )
    parser.add_argument(
        'csv_out',
        type=Path,
        help='Path to the CSV Wikidata dump to preprocess.',
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

    csv_in_path: Path = args.csv_in
    csv_out_path: Path = args.csv_out

    with (
        csv_in_path.open(mode='r', encoding='utf-8', newline='') as csv_in,
        csv_out_path.open(mode='w', encoding='utf-8', newline='') as csv_out,
    ):
        reader = csv.DictReader(csv_in)
        columns_to_drop = choose_columns_to_drop(reader.fieldnames)
        # PyCharm gets confused and thinks the rows are strings
        output_columns = [
            clean_column(column)
            for column in reader.fieldnames
            if column not in columns_to_drop
        ]
        writer = csv.DictWriter(csv_out, fieldnames=output_columns)
        writer.writeheader()
        row: dict[str, Any]
        for row in reader:
            kept_columns_only = {
                column: value
                for column, value in row.items()
                if column not in columns_to_drop
            }
            cleaned = {
                clean_column(column): value for column, value in kept_columns_only.items()
            }
            writer.writerow(cleaned)


if __name__ == '__main__':
    main()
