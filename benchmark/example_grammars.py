#_______________________________________________________________________________
# JSON grammar

# https://lark-parser.readthedocs.io/en/latest/examples/json_parser.html
json_grammar = r"""
?start: value

?value: object
      | array
      | string
      | SIGNED_NUMBER      -> number
      | "true"             -> true
      | "false"            -> false
      | "null"             -> null

array  : "[" [value ("," value)*] "]"
object : "{" [pair ("," pair)*] "}"
pair   : string ":" value

string : ESCAPED_STRING

%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS

%ignore WS
"""

#_______________________________________________________________________________
#

# https://github.com/esteng/ambiguous_parsing ZERO AND FEW-SHOT SEMANTIC PARSING
# WITH AMBIGUOUS INPUTS (Stengel-Eskin, Rawlins, and Van Durme; 2024)
# https://arxiv.org/pdf/2306.00824

amp = r"""

sent: "exists " var " . " sent
| "forall " var " . " sent
| "( " sent " )"
| sent " AND " sent
| sent " OR " sent
| expr "(" var ")"
| expr "(" var ", " var ")"
| expr "(" var ", " const ")"

var: "x" | "y" | "z" | "a" | "e" | "i"

expr: "boy" | "girl" | "man" | "woman" | "bird" | "cat" | "dog" | "fish" | "cow" | "elephant"
| "book" | "rock" | "table" | "cup" | "crayon" | "telescope" | "binoculars" | "camera" | "spyglass"
| "gloves" | "mittens" | "ovenmitts" | "pyjamas" | "pants" | "sweater" | "hat" | "saw" | "observed"
| "watched" | "spied" | "picked up" | "grabbed" | "held" | "lifted" | "heard" | "listened" | "chased"
| "followed" | "called" | "ate" | "drank" | "slept" | "walked" | "left" | "played" | "moved" | "drew"
| "napped" | "picked_up" | "agent" | "patient" | "instrument" | "have"

const: "Galileo" | "Marie" | "Sherlock" | "Ada" | "Alan" | "Katherine" | "Watson" | "Adele" | "Bill"

"""


#_______________________________________________________________________________
# Specialized SQL grammar

grammar1 = r"""
start: WS? "SELECT" WS select_expr WS "FROM" WS from_expr [WS "WHERE" WS bool_condition] [WS "GROUP BY" WS var_list] [WS "ORDER BY" WS orderby_expr] WS EOS
EOS: "</s>"
select_expr: STAR | select_list
bool_condition: bool_expr | "(" bool_condition WS "AND" WS bool_condition ")" | "(" bool_condition WS "OR" WS bool_condition ")"
bool_expr: var "=" value | var ">" value | var "<" value
from_expr: "data"
orderby_expr: var_list WS "ASC" | var_list WS "DESC"
select_list: select_var ("," WS select_var)*
var_list: var ("," WS var)*
select_var: var | "AVG(" var ")" | "MEDIAN(" var ")" | "COUNT(" var ")"
var: "age" | "gender" | "year" | "state_color" | "zipcode" | "vote" | "race_ethnicity"
value: NUMBER | "red" | "blue" | "white" | "black" | "latino" | "republican" | "democrat" | "male" | "female"
STAR: "*"
NUMBER: /\d+/
WS: /[ \t\f\r\n]/
"""

restricted_sql = r"""
start: query_expr "</s>"

query_expr: select [ "ORDER" "BY" (order_by_expr ",")*  order_by_expr] [ "LIMIT" integer_ ]

select: "SELECT" [(select_expr ",")*] select_expr "FROM" "data" [ "WHERE" bool_expression ] [ "GROUP" "BY" [(expression ",")*] expression ]

select_expr.0: expression_math [ [ "AS" ] alias ] -> select_expression

?expression_math: expression_product
               | expression_math PLUS expression_product -> expression_add
               | expression_math MINUS expression_product -> expression_sub
               | AGGREGATION expression_math /\)/ -> sql_aggregation

?expression: (name | STAR) -> column_name
            | literal

?expression_product: expression_parens
                  | expression_product STAR expression_parens -> expression_mul
                  | expression_product "/" expression_parens -> expression_div

?expression_parens: expression
                  | /\(/ expression_parens STAR expression /\)/ -> expression_mul
                  | /\(/  expression_parens "/" expression /\)/ -> expression_div
                  | /\(/  expression_parens PLUS expression /\)/ -> expression_add
                  | /\(/  expression_parens MINUS expression /\)/ -> expression_sub

bool_expression: bool_parentheses
                 | bool_expression "AND" bool_parentheses -> bool_and
                 | bool_expression "OR" bool_parentheses -> bool_or
bool_parentheses: comparison_type
                 | /\(/   bool_expression "AND" comparison_type /\)/ -> bool_and
                 | /\(/  bool_expression "OR" comparison_type /\)/ -> bool_or
comparison_type: equals | not_equals | greater_than | less_than | greater_than_or_equal
| less_than_or_equal | is_null | is_not_null
equals: expression_math "=" expression_math
not_equals: expression_math ("<>" | "!=") expression_math
greater_than: expression_math ">" expression_math
less_than: expression_math "<" expression_math
greater_than_or_equal: expression_math ">=" expression_math
less_than_or_equal: expression_math "<=" expression_math
is_null: expression_math "is" "null"
is_not_null: expression_math "is" "not" "null"

alias: /[A-Za-z]/
name: /[A-Za-z]/
PLUS: /\+/
MINUS: /[\-]/

order_by_expr: expression_math ["ASC"] -> order_asc
        | expression_math "DESC" -> order_desc

AGGREGATION.8: ("sum(" | "avg(" | "min(" | "max(" | "count(" "distinct" | "count(")
STAR: /\*/
integer_: /[1-9][0-9]*/
?literal: boolean -> bool
       | integer_ -> number
       | /'([^']|\s)+'|''/ -> string

boolean: "true" -> true
       | "false" -> false

%import common.WS
%ignore WS
"""
