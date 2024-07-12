#!/usr/bin/env bash

script_root=$(dirname $(realpath "${BASH_SOURCE[0]}"))

if [[ "$#" -ne 2 ]]; then
  printf '%s\n' "usage: $0 csv_path db_path"
fi

raw_csv_path=$1
db_path=$2

csv_path="${raw_csv_path%.csv}_preprocessed.csv"
python preprocess_wikidata.py "$raw_csv_path" "$csv_path" ||
  exit 1

export csv_path
cat "$script_root"/import_wikidata_db.sql |
    envsubst |
    sqlite3 "$db_path"
