from genparse.cfglm import EarleyBoolMaskCFGLM
from genparse.util import LarkStuff

very_restricted_sql = r"""
    start: WS? "SELECT"i WS from_expr [WS "WHERE"i WS bool_condition] [WS "GROUP BY"i WS var_list] [WS "ORDER BY"i WS orderby_expr] WS EOS
    EOS: "</s>"
    select_expr: STAR | select_list
    bool_condition: bool_expr | "(" bool_condition WS "AND"i WS bool_condition ")" | "(" bool_condition WS "OR"i WS bool_condition ")"
    bool_expr: var "=" value | var ">" value | var "<" value
    from_expr: "data"
    orderby_expr: var_list WS "ASC"i | var_list WS "DESC"i
    select_list: select_var ("," WS select_var)*
    var_list: var ("," WS var)*
    select_var: var | "AVG"i "(" var ")" | "MEDIAN"i "(" var ")" | "COUNT"i "(" var ")"
    var: "age" | "gender" | "year" | "state_color" | "zipcode" | "vote" | "race_ethnicity"
    value: NUMBER | "'red'" | "'blue'" | "'white'" | "'black'" | "'latino'" | "'republican'" | "'democrat'" | "'male'" | "'female'"
    STAR: "*"
    NUMBER: /\d+/
    WS: /[ ]/
"""

guide = EarleyBoolMaskCFGLM(LarkStuff(very_restricted_sql).char_cfg(0.99, ignore="[ ]?"))
assert set(guide.p_next("").keys()) == {"S","s"," "}
assert set(guide.p_next("S").keys()) == {"E","e"}
assert set(guide.p_next("s").keys()) == {"E","e"}
