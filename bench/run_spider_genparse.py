#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate spider on vLLM Llama-3 with grammar restriction.

Example usage:
CUDA_VISIBLE_DEVICES=0 python benchmark/run_spider_genparse.py --particles 20 --n-beam 20 --n-query 1000 --inference smc-standard
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path

import transformers
from tqdm import tqdm

from genparse.lm import TokenizedLLM
from genparse.util import lark_guide

# from genparse.experimental.steer_local import LocalPOESampler
from genparse.backends.vllm import vllmpplLLM, VLLMSampler
from genparse.proposal import CharacterProposal, TokenProposal
from genparse.util import set_seed
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
    parser.add_argument('--exp-name', type=str, default='llama3-8b')
    parser.add_argument(
        '--inference',
        choices=[
            'smc-standard',
            #'smc-steer',
            'importance-sampling',
            'local-poe',
        ],
        default='smc-standard',
    )
    parser.add_argument('--particles', type=int, default=1)
    parser.add_argument('--n-beam', type=int, default=1)  # XXX: no longer used.
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument(
        '--proposal',
        choices=['character', 'token'],
        default='character',
        help='Specify which proposal distribution to use in SMC inference.',
    )
    parser.add_argument(
        '--decision-rule',
        choices=['mbr', 'viterbi'],
        default='mbr',
        help='Specify a decision rule for selecting a query from posterior estimate',
    )
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--K',
        type=int,
        default=20,
        help='parameter for token proposal distribution',
    )
    parser.add_argument(
        '--schema-grammar', action='store_true', help='use schema-specific grammar'
    )

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
    )  # Dict[str, DbSchema], schema_name -> schema

    spider_dev_data = load_spider_data(raw_spider_dir / 'dev.json')
    spider_train_data = load_spider_data(raw_spider_dir / 'train_spider.json')
    logger.info('spider data loaded.')

    prompt_formatter = SpiderPromptFormatter(spider_train_data, spider_schemas)
    evaluator = Evaluator(raw_spider_dir)

    # Initialize model.
    logger.info(f'Initializing model: {args.model_name} ...')
    hfppl_llm = vllmpplLLM(args.model_name, max_model_len=4096, seed=args.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)

    BATCH_SIZE = 80

    hfppl_llm.batch_size = BATCH_SIZE
    genparse_llm = TokenizedLLM(
        model=hfppl_llm, tokenizer=tokenizer, batch_size=BATCH_SIZE
    )

    guides = {}  # schema_name -> guide,
    # `guides` is only used in schema-specific to store asynchronously constructed
    # parsers to save time
    samplers = {}  # schema_name -> (sampler, proposal)

    pool = mp.Pool(os.cpu_count() // 2 - 1)  # half cpus per gpu and leave one for main process

    if not args.schema_grammar:  # use the permissive general sql grammar
        grammar_file = 'benchmark/grammars/sql_case_insensitive.lark'
        print(f'using grammar from: {grammar_file}')
        with open(grammar_file, 'r') as f:
            grammar = f.read()
        guide = lark_guide(grammar)
        sampler = VLLMSampler(llm=genparse_llm, guide=guide)
        if args.proposal == 'character':
            proposal = CharacterProposal(llm=genparse_llm, guide=guide)
        else:
            assert args.proposal == 'token'
            proposal = TokenProposal(llm=genparse_llm, guide=guide, K=args.K)

        for schema_name in spider_schemas:
            samplers[schema_name] = (sampler, proposal)

    else:  # schema-specific grammars
        grammar_file = 'spider_schema_grammar.json'
        print(f'using schema-specific grammar file from: {grammar_file}')
        with open(grammar_file, 'r') as f:
            all_grammars = json.load(f)

        # re-order the list of the grammars so that they're in the order they appear
        # in the dataset. Use dict over set because python guarantees insertion order in
        # dict, not in set.
        reordered_grammar_names = {}
        for dev_datum in spider_dev_data:
            if dev_datum.schema_name not in reordered_grammar_names:
                reordered_grammar_names[dev_datum.schema_name] = None

        for schema_name in reordered_grammar_names:
            grammar = all_grammars[schema_name]
            grammar = reformat_grammar(grammar)
            # guide = lark_guide(grammar)
            guides[schema_name] = pool.apply_async(lark_guide, (grammar,))

    logger.info('Model(s) initialized (asynchronously when schema specific).')

    # Setup saving.
    n_query = args.n_query

    outpath = f'{args.exp_name}-{args.inference}-p{args.particles}-b{args.n_beam}-{n_query}'
    if args.schema_grammar:
        outpath += '-schema'
    outpath += '.jsonl'
    outfile = open(outpath, 'w+')
    logger.info(f'writing to {outpath} ... ')

    n_correct, n_invalid, n_mismatch = 0, 0, 0

    for i, dev_datum in tqdm(enumerate(spider_dev_data[:n_query]), total=n_query):

        messages = prompt_formatter.format_openai(dev_datum)

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

        if dev_datum.schema_name in samplers:
            # if not schema specific, these are ready by default
            # if schema specific, the parser is async parser has been used and is ready
            sampler, proposal = samplers[dev_datum.schema_name]
        else:
            guide = guides[dev_datum.schema_name].get()
            sampler = VLLMSampler(llm=genparse_llm, guide=guide)
            if args.proposal == 'character':
                proposal = CharacterProposal(llm=genparse_llm, guide=guide)
            else:
                assert args.proposal == 'token'
                proposal = TokenProposal(llm=genparse_llm, guide=guide, K=args.K)
            samplers[dev_datum.schema_name] = (sampler, proposal)

        particles = sampler.run_inference(
            prompt=prompt,
            proposal=proposal,
            method=args.inference,
            n_particles=args.particles,
            max_tokens=args.max_tokens,
            n_beam=args.n_beam,
            verbosity=args.verbosity,
        )

        if args.decision_rule == 'mbr':
            # Minimum Bayes risk decision picks the particle with the highest
            # expected `match` under the posterior approximation.

            def match(x, y):
                x = x.rstrip(guide.eos)
                y = y.rstrip(guide.eos)
                (exec_match, _) = evaluator.evaluate(x, y, db_name=dev_datum.schema_name)
                return exec_match

            # from arsenal import timeit
            # with timeit('mbr'):
            pmax = max(
                particles,
                key=lambda candidate: particles.risk(match, ''.join(candidate.context)),
            )

        else:
            assert args.decision_rule == 'viterbi'

            # Viterbi decision picks the highest-weight particle under the
            # posterior approximation.

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
        else:
            raise ValueError

        n_total = sum((n_correct, n_invalid, n_mismatch))
        print(
            f'correct: {n_correct / n_total:.2f} ({n_correct}), '
            f'invalid: {n_invalid / n_total:.2f} ({n_invalid}), '
            f'mismatch: {n_mismatch / n_total:.2f} ({n_mismatch})'
        )

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
