import json
from genparse.cfg import CFG, Rule
from genparse.semiring import Real


def convert_rules(benchclamp_rule_set):
    lhs, rhs_list = benchclamp_rule_set
    return [
        Rule(
            Real(1),  # weight
            'nt_' + lhs,
            convert_rhs(rhs),
        )
        for rhs in rhs_list
    ]


def convert_rhs_token(token):
    print(token)

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
        R=Real,
        S='nt_start',
        V=set(terminals),
    )

    for rule in rules:
        cfg.add(rule.w, rule.head, *rule.body)

    return cfg


if __name__ == '__main__':
    grammars = json.load(open('benchmark/grammars/benchclamp_spider_grammars.json', 'r'))
    grammar = grammars['perpetrator']
    rules = [r for rule in grammar.items() for r in convert_rules(rule)]
    cfg = make_cfg_from_rules(rules)
