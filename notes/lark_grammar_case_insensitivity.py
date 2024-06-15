from genparse.cfglm import EarleyBoolMaskCFGLM
from genparse.util import LarkStuff

with open('../benchmark/grammars/sql_case_insensitive.lark', 'r') as f:
    guide = EarleyBoolMaskCFGLM(LarkStuff(f.read()).char_cfg(0.99, ignore='[ ]?'))

assert set(guide.p_next('').keys()) == {'S', 's'}
assert set(guide.p_next('S').keys()) == {'E', 'e'}
assert set(guide.p_next('s').keys()) == {'E', 'e'}
