#!/usr/bin/env bash

script_root=$(dirname $(realpath "${BASH_SOURCE[0]}"))

if [[ "$#" -ne 1 ]]; then
  printf '%s\n' "usage: $0 db_path"
fi

db_path=$1


cat "$script_root"/setup_toy_db.sql |
    sqlite3 "$db_path"
