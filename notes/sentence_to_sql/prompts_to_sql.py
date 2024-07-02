"""
Converts prompts into SQL queries about the people/orgs in those prompts.
"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Any

import transformers

from genparse.cfglm import BoolCFGLM
from genparse.lm import AsyncGreedilyTokenizedLLM
from genparse.proposal import CharacterProposal
from genparse.vllm_compatibility import vllmpplLLM
from genparse.vllm_steer import VLLMSampler
from genparse.util import LarkStuff

logger = logging.getLogger(__name__)
repo_root = Path(__file__).resolve().parent.parent.parent

_SQL_GRAMMAR = r"""
// Adapted from https://github.com/probcomp/genparse/blob/main/benchmark/grammars/sql_case_insensitive.lark
// Adapted from https://github.com/zbrookle/sql_to_ibis and https://github.com/lapp0/outlines 
// License for https://github.com/zbrookle/sql_to_ibis follows
//BSD 3-Clause License
//
//Copyright (c) 2011-2022, Open source contributors.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//
//* Redistributions of source code must retain the above copyright notice, this
//  list of conditions and the following disclaimer.
//
//* Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
//
//* Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


start: set_expr

set_expr: query_expr
        | set_expr set_op set_expr
        | set_expr set_op set_op_case set_expr

set_op: "UNION"i | "INTERSECT"i | "EXCEPT"i
set_op_case: "ALL"i | "DISTINCT"i

query_expr: select [ "ORDER"i "BY"i (order_by_expr ",")*  order_by_expr] [ "LIMIT"i limit_count [ "OFFSET"i skip_rows ] ]

select: "SELECT"i [SELECT_CONSTRAINT] [(select_expr ",")*] select_expr "FROM"i [(from_expr ",")*] from_expr [ "WHERE"i where_expr ] [ "GROUP"i "BY"i [(groupby_expr ",")*] groupby_expr ] [ "HAVING"i having_expr] [ "WINDOW"i window_expr ]

where_expr: bool_expression | name "LIKE"i literal //ADD LIKE CLAUSES

select_expr.0: expression_math [ [ "AS"i ] alias ] -> select_expression

?from_expr: from_item -> from_expression

order_by_expr: order -> order_by_expression

having_expr: bool_expression

groupby_expr: expression -> group_by

window_expr: [window_expr ","] _window_name "AS"i ( window_definition )

// jac: s/name/table_name/
from_item: table_name [ [ "AS"i ] alias ] -> table
            | join -> join
            | cross_join -> cross_join_expression
            | subquery
// new rule: table_name?
?table_name: "People"
subquery: ( "(" (set_expr | join | cross_join) ")" ) [ [ "AS"i ] alias ]

cross_join: from_item "CROSS"i "JOIN"i from_item
join: from_item JOIN_EXPR from_item [ "ON"i bool_expression ] -> join_expression

JOIN_EXPR.5: (JOIN_TYPE WS)? "JOIN"i
JOIN_TYPE: "INNER"i | "OUTER"i? | JOIN_DIRECTION (WS "OUTER"i)? | JOIN_DIRECTION
JOIN_DIRECTION: "FULL"i | "LEFT"i | "RIGHT"i

?expression_math: expression_product
               | expression_math "+" expression_product -> expression_add
               | expression_math MINUS expression_product -> expression_sub
               | "CASE"i (when_then)+ "ELSE"i expression_math "END"i -> case_expression
               | "CAST"i" (" expression_math "AS"i TYPENAME ")" -> as_type
               | "CAST"i" (" literal "AS"i TYPENAME ")" -> literal_cast
               //| AGGREGATION /\s?\(/ expression_math /\)\s?/ -> sql_aggregation 
               | AGGREGATION expression_math /\s?\)/ [window_form] -> sql_aggregation
               | "RANK"i "(" ")" window_form -> rank_expression
               | "DENSE_RANK"i "(" ")" window_form -> dense_rank_expression
               | "COALESCE"i "(" [(expression_math ",")*] expression_math ")" -> coalesce_expression
               | subquery // TOO PERMISSIVE????????

window_form: "OVER"i "(" ["PARTITION"i "BY"i (partition_by ",")* partition_by] ["ORDER"i "BY"i (order ",")* order [ row_range_clause ] ] ")"

partition_by: expression_math

row_range_clause: ( ROWS | RANGE ) frame_extent
frame_extent: frame_between | frame_preceding
frame_between: "BETWEEN"i frame_bound "AND"i frame_bound
frame_bound: frame_preceding | frame_following | "CURRENT"i "ROW"i
frame_preceding: UNBOUNDED PRECEDING | integer_ PRECEDING
frame_following: UNBOUNDED FOLLOWING | integer_ FOLLOWING
RANGE: "RANGE"i
ROWS: "ROWS"i
UNBOUNDED: "UNBOUNDED"i
PRECEDING: "PRECEDING"i
FOLLOWING: "FOLLOWING"i

when_then: "WHEN"i bool_expression "THEN"i expression_math
order: expression_math ["ASC"i] -> order_asc
          | expression_math "DESC"i-> order_desc

// jac: modified to include only the columns actually appearing in Ian L's People data from Wikidata
// column_name: [name "."] name
column_name: valid_column_name
valid_column_name: _common_name | _given_name | _family_name | _name_in_native_language | _languages_spoken_written_or_signed | _date_of_birth | _place_of_birth | _spouse | _mother | _father | _country_of_citizenship | _occupation | _religion_or_worldview | _sex_or_gender
_common_name: "itemLabel.value"
_given_name: "given_nameLabel.value"
_family_name: "family_nameLabel.value"
_name_in_native_language: "name_in_native_languageLabel.value"
_languages_spoken_written_or_signed: "languages_spoken__written_or_signedLabel.value"
_date_of_birth: "date_of_birthLabel.value"
_place_of_birth: "place_of_birthLabel.value"
_spouse: "spouseLabel.value"
_mother: "motherLabel.value"
_father: "fatherLabel.value"
_country_of_citizenship: "country_of_citizenshipLabel.value"
_occupation: "occupationLabel.value"
_religion_or_worldview: "religion_or_worldviewLabel.value"
_sex_or_gender: "sex_or_genderLabel.value"
// jac: end of changed bit
?expression_product: expression_parens 
                  | expression_product "*" expression_parens -> expression_mul
                  | expression_product "/" expression_parens -> expression_div

?expression_parens: expression
                  | "(" expression_parens "*" expression ")" -> expression_mul
                  | "(" expression_parens "/" expression ")" -> expression_div
                  | "(" expression_parens "+" expression ")" -> expression_add
                  | "(" expression_parens MINUS expression ")" -> expression_sub

// jac: modified rule to only allow valid column names
// ?expression: [name "."] (name | STAR) -> column_name
?expression: (valid_column_name | STAR) -> column_name
            | literal


SELECT_CONSTRAINT.9: "ALL"i | "DISTINCT"i
TYPENAME:  "object"
         | "varchar"
         | "integer"
         | "int16"
         | "smallint"
         | "int32"
         | "int64"
         | "int"
         | "bigint"
         | "float16"
         | "float32"
         | "float64"
         | "float"
         | "bool"
         | "datetime64"
         | "timestamp"
         | "time"
         | "date"
         | "category"
         | "string"
AGGREGATION: /(sum|SUM)\s?\(/ | /(avg|AVG)\s?\(/ | /(min|MIN)\s?\(/ | /(max|MAX)\s?\(/ | "count"i /\s?\(\s?(distinct|DISTINCT)/ | /(count|COUNT)\s?\(/
alias: name -> alias_string
_window_name: name
limit_count: integer_ -> limit_count
skip_rows: integer_
bool_expression: bool_parentheses
                 | bool_expression "AND"i bool_parentheses -> bool_and
                 | bool_expression "OR"i bool_parentheses -> bool_or
bool_parentheses: comparison_type
                 | "(" bool_expression "AND"i comparison_type ")" -> bool_and
                 | "(" bool_expression "OR"i comparison_type ")" -> bool_or
comparison_type: equals | not_equals | greater_than | less_than | greater_than_or_equal | less_than_or_equal | between | in_expr | not_in_expr | subquery_in | is_null | is_not_null // ADD SUBQUERY
equals: expression_math "=" expression_math
is_null: expression_math "is""null"
is_not_null: expression_math "is"i "not"i "null"i
not_equals: expression_math ("<>" | "!=") expression_math
greater_than: expression_math ">" expression_math
less_than: expression_math "<" expression_math
greater_than_or_equal: expression_math ">=" expression_math
less_than_or_equal: expression_math "<=" expression_math
between: expression_math "BETWEEN"i expression_math "AND"i expression_math
in_expr: expression_math "IN"i "(" [expression_math ","]* expression_math ")"
subquery_in: expression_math "IN"i subquery
//not_in_expr: expression_math "NOT"i "IN"i "(" [expression_math ","]* expression_math ")"
not_in_expr: expression_math "NOT"i "IN"i subquery // TOO RESTRICTIVE (subquery are possible expression_math terms)??
?literal: boolean -> bool
       | number_expr -> number
       | ESCAPED_STRING -> string
       | timestamp_expression -> timestamp_expression
boolean: "true"-> true
       | "false"-> false
?number_expr: product

?product: NUMBER

integer_: /[1-9][0-9]*/
STAR: "*"
window_definition:
timestamp_expression: "NOW"i "(" ")" -> datetime_now
                    | "TODAY"i "(" ")" -> date_today
              //       | "TIMESTAMP"i "(" "'" date "'" "," "'" time "'" ")" -> custom_timestamp

date: YEAR MINUS MONTH MINUS DAY
YEAR: /[0-9]{4}/
MONTH: /[0-9]{2}/
DAY: /[0-9]{2}/
time: HOURS ":" MINUTES ":" SECONDS
HOURS: /[0-9]{2}/
MINUTES: /[0-9]{2}/
SECONDS: /[0-9]{2}/
name: CNAME | ESCAPED_STRING
MINUS: /-/

LETTER: "A".."Z" | "a".."z"
DIGIT: "0".."9"
SQUOTE: "'"
DQUOTE: "\""
BACKSLASH: "\\"
SYMBOL: "(" | ")" | "[" | "]" | "{" | "}" | "!" | "@" | "%" | "*" | "." | "," | ":" | ";" | "<" | "=" | ">" | "/" | "?" | "_" | "`" | "\\"
CHAR_INNER: LETTER|DIGIT|SYMBOL|WS
STRING_INNER_S: (CHAR_INNER|DQUOTE|(BACKSLASH SQUOTE))*
STRING_INNER_D: (CHAR_INNER|SQUOTE|(BACKSLASH DQUOTE))*
ESCAPED_STRING: SQUOTE STRING_INNER_S SQUOTE | DQUOTE STRING_INNER_D DQUOTE
INT: DIGIT+
FLOAT: INT "." INT? | "." INT
NUMBER: FLOAT | INT

%import common.CNAME
%import common.WS
%import common.SQL_COMMENT
%import common.WS_INLINE

%ignore WS
%ignore SQL_COMMENT
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(mode='r', encoding='utf-8') as jsonl_in:
        result = [json.loads(line) for line in jsonl_in if line.strip()]
    return result


def save_jsonl(outputs: list[dict[str, Any]], path: Path) -> None:
    with path.open(mode='w', encoding='utf-8') as jsonl_out:
        for output in outputs:
            json.dump(output, jsonl_out)
            jsonl_out.write('\n')


def run_inference(
    model_name: str,
    prompts: list[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    n_particles: int,
) -> list[dict[str, float]]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    llm = AsyncGreedilyTokenizedLLM(
        model=vllmpplLLM(model_name),
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    guide = BoolCFGLM(LarkStuff(_SQL_GRAMMAR).char_cfg(0.99, ignore='[ ]?'))
    proposal = CharacterProposal(llm=llm, guide=guide)
    sampler = VLLMSampler(llm=llm, guide=guide)
    result = [
        sampler.run_inference(
            prompt=prompt,
            proposal=proposal,
            method='smc-standard',
            max_tokens=max_new_tokens,
            n_particles=n_particles,
            verbosity=0,
        ).posterior
        for prompt in prompts
    ]
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('prompts_path', type=Path, help='Path to the prompts file.')
    parser.add_argument(
        'save_inferences_to',
        type=Path,
        help='Where to save the JSONL file of inferences.',
    )
    parser.add_argument(
        '--model', default='gpt2', help='The language model to use for inference.'
    )
    parser.add_argument(
        '--batch-size', type=int, default=10, help='The batch size to use for sampling.'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=128,
        help='The maximum number of tokens to generate',
    )
    parser.add_argument(
        '--n-particles',
        type=int,
        default=15,
        help='The number of particles to use to approximate the posterior.',
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

    prompts_path: Path = args.prompts_path
    save_inferences_to: Path = args.save_inferences_to
    model: str = args.model
    batch_size: int = args.batch_size
    max_new_tokens: int = args.max_new_tokens
    n_particles: int = args.n_particles

    prompt_dicts = load_jsonl(prompts_path)
    inferences = run_inference(
        model,
        [prompt_dict['prompt'] for prompt_dict in prompt_dicts],
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        n_particles=n_particles,
    )
    outputs = [
        {
            **prompt_dict,
            'genparse_inference': inference_dict,
            'inference_metadata': {
                'model': model,
                'batch_size': batch_size,
                'n_particles': n_particles,
            },
        }
        for prompt_dict, inference_dict in zip(prompt_dicts, inferences)
    ]
    save_jsonl(outputs, save_inferences_to)


if __name__ == '__main__':
    main()
