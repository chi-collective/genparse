#!/bin/bash

set -e

cwd=$(pwd)

cd "$cwd"/amp
python -m pip install -e .
# pytest test

# cd "$cwd"/amp/grammar
# python create_grammar.py --base_path "$cwd"/amp/grammar/cfgs/lark/base_lark.cfg --out_path "$cwd"/amp/grammar/cfgs/lark/lark.cfg
# python check_coverage.py -n 100
