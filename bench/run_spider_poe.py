#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate spider on vLLM Llama-3 with grammar restriction.

Example usage:
CUDA_VISIBLE_DEVICES=0 python benchmark/run_spider_genparse.py --particles 20 --n-beam 20 --n-query 1000 --inference smc-standard
"""

import argparse
import json
import logging
import os
from pathlib import Path

import transformers
from tqdm import tqdm

from genparse.cfglm import BoolCFGLM
from genparse.lm import TokenizedLLM
from genparse.util import lark_guide
from genparse.experimental.steer_local import LocalPOESampler
from genparse.backends.vllm import vllmpplLLM, VLLMSampler
from genparse.proposal import CharacterProposal
from genparse.util import LarkStuff, set_seed
from bench.spider.dialogue import load_spider_data
from bench.spider.evaluator import Evaluator
from bench.spider.schema import load_schemas
from bench.spider.prompt_formatter import SpiderPromptFormatter

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-query', type=int, default=100)
    parser.add_argument(
        '--model-name',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        choices=['meta-llama/Meta-Llama-3-8B-Instruct'],
    )
    parser.add_argument('--exp-name', type=str, default='llama3-8b-100')
    parser.add_argument(
        '--inference',
        choices=['smc-standard', 'smc-steer', 'importance-sampling', 'local-poe'],
        default='smc-standard',
    )
    parser.add_argument('--particles', type=int, default=1)
    parser.add_argument('--n-beam', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--K', type=int, default=20)

    return parser


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


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )
    # disable unnecessary logs from httpx used by openai client
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    # for easy running from cli
    access_token = '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'
    os.environ['HF_TOKEN'] = access_token

    # Load data.
    logger.info('loading spider data...')
    raw_spider_dir = Path('bench/spider/data/spider')
    spider_schemas = load_schemas(
        schemas_path=raw_spider_dir / 'tables.json', db_path=raw_spider_dir / 'database'
    )

    spider_dev_data = load_spider_data(raw_spider_dir / 'dev.json')
    spider_train_data = load_spider_data(raw_spider_dir / 'train_spider.json')
    logger.info('spider data loaded.')

    prompt_formatter = SpiderPromptFormatter(spider_train_data, spider_schemas)
    evaluator = Evaluator(raw_spider_dir)

    # Initialize model.

    BATCH_SIZE = 80

    logger.info(f'Initializing model: {args.model_name} ...')

    if args.inference == 'local-poe':
        logger.info('Using local product of experts')
        from vllm import LLM

        vllm_llm = LLM(model=args.model_name)
        tokenizer = vllm_llm.get_tokenizer()
    else:
        hfppl_llm = vllmpplLLM(args.model_name, max_model_len=4096, seed=args.seed)
        hfppl_llm.batch_size = BATCH_SIZE
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
        with open('benchmark/grammars/sql_case_insensitive.lark', 'r') as f:
            grammar = f.read()
        genparse_llm = TokenizedLLM(
            model=hfppl_llm, tokenizer=tokenizer, batch_size=BATCH_SIZE
        )
        guide = BoolCFGLM(LarkStuff(grammar).char_cfg(0.99))
        sampler = VLLMSampler(llm=genparse_llm, guide=guide)
        proposal = CharacterProposal(llm=genparse_llm, guide=guide)

    logger.info('Model initialized.')

    # Setup saving.
    n_query = args.n_query
    outpath = f'llama-3-{args.inference}-p{args.particles}-b{args.n_beam}-{n_query}.jsonl'
    outfile = open(outpath, 'w+')
    logger.info(f'writing to {outpath} ... ')

    n_correct, n_invalid, n_mismatch = 0, 0, 0

    guides = {}

    grammars = json.load(open('benchmark/grammars/spider_schema_grammar.json', 'r'))

    for i, dev_datum in tqdm(enumerate(spider_dev_data[:n_query]), total=n_query):
        messages = prompt_formatter.format_openai(dev_datum)

        if dev_datum.schema_name in guides:
            guide = guides[dev_datum.schema_name]
        else:
            if dev_datum.schema_name in grammars:
                guide = lark_guide(reformat_grammar(grammars[dev_datum.schema_name]))
                guides[dev_datum.schema_name] = guide
            else:
                print(f'skipping {dev_datum.schema_name}')

        if i == 0:  # print an example for demonstration
            print('=' * 30 + ' Example prompt ' + '=' * 30)
            for msg in messages:
                print(msg['role'] + ':')
                print('=' * (len(msg['role']) + 1))
                print(msg['content'])
                print('-' * 100)
            print('=' * 30 + '  End of prompt ' + '=' * 30)

        prompt = tokenizer.decode(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        )

        if args.inference == 'local-poe':
            sampler = LocalPOESampler(vllm_llm, guide, K=args.K)
            particles = sampler.run_inference(
                prompt=prompt,
                n_particles=args.particles,
                max_tokens=args.max_tokens,
                seed=args.seed,
            )
        else:
            particles = sampler.run_inference(
                prompt=prompt,
                proposal=proposal,
                method=args.inference,
                n_particles=args.particles,
                max_tokens=args.max_tokens,
                n_beam=args.n_beam,
                verbosity=args.verbosity,
            )

        pmax = particles.particles[0]
        for p in particles.particles[1:]:
            if p.finished and p.weight > pmax.weight:
                pmax = p

        particles_json = [
            {
                'tokens': p.context,
                'token_ids': p.context_ids,
                'weight': p.weight,
                'finished': p.finished,
            }
            for p in particles
        ]

        pred = ''.join(pmax.context[:-1])
        gold = dev_datum.query
        db = dev_datum.schema_name
        result = evaluator.evaluate(gold, pred, db)

        result_s = json.dumps(
            {
                'pred': pred,
                'gold': gold,
                'db_name': db,
                'question': dev_datum.utterance,
                'result': result,
                'finished': pmax.finished,
                'tokens': pmax.context,
                'token_ids': pmax.context_ids,
                'particles': particles_json,
            }
        )
        print(pred)
        print(result)
        print(result_s, file=outfile)

        if result[0]:
            n_correct += 1
        elif result[1] == 'invalid':
            n_invalid += 1
        elif result[1] == 'mismatch':
            n_mismatch += 1

        print(
            f'correct: {n_correct / (i + 1):.2f}, '
            f'invalid: {n_invalid / (i + 1):.2f}, '
            f'mismatch: {n_mismatch / (i + 1):.2f}'
        )


if __name__ == '__main__':
    main()
