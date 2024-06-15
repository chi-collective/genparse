# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

DOMAINS_DIR = Path(__file__)

# TODO: Don't hardcode
COSQL_DIR = Path('data/cosql_dataset')

SCHEMAS_FILE = COSQL_DIR / 'tables.json'

SQL_STATE_TRACKING_DIR = COSQL_DIR / 'sql_state_tracking'
TRAIN_DATA_FILE = SQL_STATE_TRACKING_DIR / 'cosql_train.json'

SQL_GRAMMAR_DIR = DOMAINS_DIR / 'grammar'
