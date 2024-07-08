import json
from pathlib import Path

from bench.spider.dialogue import load_spider_data
from bench.spider.schema import load_schemas
from genparse.cfg import CFG, Rule
from genparse.semiring import Float
from genparse.cfglm import BoolCFGLM


def convert_rules(benchclamp_rule_set):
    lhs, rhs_list = benchclamp_rule_set
    return [
        Rule(
            1.0,  # weight
            'nt_' + lhs,
            convert_rhs(rhs),
        )
        for rhs in rhs_list
    ]


def convert_rhs_token(token):
    print(token)

    if token["type"] == "terminal":
        name = json.loads(token['underlying'])
    else:
        name = token['underlying']
    if token['optional']:
        # haven't coded this yet, assuming this doesn't happen in the grammars
        raise NotImplementedError
    if token['type'] == 'nonterminal':
        name = 'nt_' + name
    return name


def convert_rhs(benchclamp_rhs):
    if benchclamp_rhs == [{}]:
        return ()
    return tuple(convert_rhs_token(token) for token in benchclamp_rhs)


def make_cfg_from_rules(rules):
    terminals = [s for r in rules for s in r.body if s[:3] != 'nt_']

    cfg = CFG(
        R=Float,
        S='nt_start',
        V=set(terminals),
    )

    for rule in rules:
        cfg.add(rule.w, rule.head, *rule.body)

    return cfg


if __name__ == '__main__':
    grammars = json.load(open('benchmark/grammars/benchclamp_spider_grammars.json', 'r'))
    grammar = grammars['concert_singer']
    rules = [r for rule in grammar.items() for r in convert_rules(rule)]
    cfg = make_cfg_from_rules(rules)
    guide = BoolCFGLM(cfg, alg='earley')
    raw_spider_dir = Path('bench/spider/data/spider')
    spider_schemas = load_schemas(
        schemas_path=raw_spider_dir / 'tables.json', db_path=raw_spider_dir / 'database'
    )

    spider_dev_data = load_spider_data(raw_spider_dir / 'dev.json')

    result = guide.p_next(spider_dev_data[0].query)
    print(result)
