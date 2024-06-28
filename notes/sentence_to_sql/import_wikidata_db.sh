#!/usr/bin/env bash

script_root=$(dirname $(realpath "${BASH_SOURCE[0]}"))

if [[ "$#" -ne 2 ]]; then
  printf '%s\n' "usage: $0 csv_path db_path"
fi

csv_path=$1
db_path=$2

export csv_path
cat "$script_root"/import_wikidata_db.sql |
    envsubst |
    sqlite3 "$db_path"
