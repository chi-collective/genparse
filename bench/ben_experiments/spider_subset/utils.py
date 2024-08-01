HF_ACCESS_TOKEN = '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'


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
