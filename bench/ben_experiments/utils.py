HF_ACCESS_TOKEN = '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'


def mbr_eval(particles, evaluator, gold, db, eos):
    def match(x, y):
        x = x.rstrip(eos)
        y = y.rstrip(eos)
        try:
            (exec_match, _) = evaluator.evaluate(x, y, db_name=db)
        except Exception:
            exec_match = False
        return exec_match

    pmax = max(
        particles,
        key=lambda candidate: particles.risk(match, ''.join(candidate.context)),
    )

    pred = ''.join(pmax.context[:-1])

    return {
        'result': evaluator.evaluate(gold, pred, db),
        'pred': pred,
        'finished': pmax.done,
        'tokens': pmax.context,
        'token_ids': pmax.context_ids,
    }


def viterbi_eval(particles, evaluator, gold, db, eos):
    pmax = particles.particles[0]
    for p in particles.particles[1:]:
        if p.done and p.log_weight > pmax.log_weight:
            pmax = p

    pred = ''.join(pmax.context).rstrip(eos)

    return {
        'result': evaluator.evaluate(gold, pred, db),
        'pred': pred,
        'finished': pmax.done,
        'tokens': pmax.context,
        'token_ids': pmax.context_ids,
    }


def posterior_weighted_eval(particles, evaluator, gold, db, eos):
    weighted_acc = 0
    particle_results = {}
    for pred, p in particles.posterior.items():
        pred = pred.rstrip(eos)
        acc = evaluator.evaluate(gold, pred, db)
        assert pred not in particle_results, pred
        particle_results[pred] = acc
        weighted_acc += p * acc[0]

    return {'result': weighted_acc, 'particle_results': particle_results}


def reformat_grammar(grammar):
    """move start rule and remove zero-width rules"""
    lines = grammar.split('\n')
    new_grammar = ''
    for line in lines:
        if line == '|""i':
            continue
        if line.startswith('start'):
            new_grammar = line + '\n' + new_grammar
        else:
            new_grammar += line + '\n'

    return new_grammar


def load_spider_schemas(raw_spider_dir):
    from bench.spider.schema import load_schemas

    spider_schemas = load_schemas(
        schemas_path=raw_spider_dir / 'tables.json', db_path=raw_spider_dir / 'database'
    )
    return spider_schemas


def load_spider_data(raw_spider_dir, split):
    from bench.spider.dialogue import load_spider_data

    if split == 'train':
        return load_spider_data(raw_spider_dir / 'train_spider.json')
    elif split == 'dev':
        return load_spider_data(raw_spider_dir / 'dev.json')
    else:
        raise ValueError(f'Invalid split: {split}')


def load_evaluator(raw_spider_dir):
    from bench.spider.evaluator import Evaluator

    return Evaluator(raw_spider_dir)


def load_prompt_formatter(raw_spider_dir):
    from bench.spider.prompt_formatter import SpiderPromptFormatter

    train_data = load_spider_data(raw_spider_dir, split='train')
    spider_schemas = load_spider_schemas(raw_spider_dir)

    return SpiderPromptFormatter(train_data, spider_schemas)
