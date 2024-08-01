#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import logging
import argparse
from tqdm import tqdm
from pathlib import Path

from bench.cache import ProposalCache
from bench.spider.evaluator import Evaluator
from utils import (
    load_spider_data,
    load_prompt_formatter,
    reformat_grammar,
    HF_ACCESS_TOKEN,
)

from genparse.util import set_seed
from genparse.experimental.batch_inference import BatchVLLM, BatchStepModel, smc

os.environ['TOKENIZERS_PARALLELISM'] = '(true | false)'

logger = logging.getLogger(__name__)


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model-name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct'
    )
    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--particles', type=int, default=1)
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
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--K',
        type=int,
        default=0,
        help='parameter for token proposal distribution',
    )
    parser.add_argument('--n-processes', type=int, default=8)
    parser.add_argument('--out-dir', default='', type=str)
    parser.add_argument(
        '--schema', type=str, help='Schema to evaluat, seperated by comma'
    )

    return parser


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )

    parser = get_argparser()
    args = parser.parse_args()

    outpath = os.path.join(
        args.out_dir, f'{args.exp_name}-p{args.particles}-{args.proposal}'
    )

    if args.K != 0:
        outpath += f'-K{args.K}'

    json.dump(vars(args), open(f'{outpath}-args.json', 'w'), indent=4)

    outpath += '.jsonl'
    outfile = open(outpath, 'w+')

    set_seed(args.seed)

    raw_spider_dir = Path('../../spider/data/spider')
    spider_dev_data = load_spider_data(raw_spider_dir, split='dev')
    evaluator = Evaluator(raw_spider_dir)
    prompt_formatter = load_prompt_formatter(raw_spider_dir)

    os.environ['HF_TOKEN'] = HF_ACCESS_TOKEN
    batch_llm = BatchVLLM.from_name(args.model_name)
    tokenizer = batch_llm.get_tokenizer()
    proposal_cache = ProposalCache('guide_cache.pkl', 1)

    with open('../../../benchmark/grammars/spider_schema_grammar.json', 'r') as f:
        all_grammars = json.load(f)

    schema = args.schema.split(',')

    n_correct, n_invalid, n_mismatch = 0, 0, 0

    for i, dev_datum in tqdm(
        enumerate(spider_dev_data), total=len(spider_dev_data), smoothing=0.0
    ):
        if dev_datum.schema_name not in schema:
            continue

        messages = prompt_formatter.format_openai(dev_datum)

        grammar = reformat_grammar(all_grammars[dev_datum.schema_name])

        parallel_proposal = proposal_cache.fetch_or_create_proposal(
            llm=batch_llm.llm,
            grammar=grammar,
            proposal_name=args.proposal,
            n_processes=args.n_processes,
        )

        step_model = BatchStepModel(
            batch_proposal=parallel_proposal,
            batch_llm=batch_llm,
            max_tokens=args.max_tokens,
        )

        prompt = tokenizer.decode(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        )

        step_model.set_prompt(prompt)

        start_time = time.time()

        particles = smc(step_model, n_particles=args.particles)

        end_time = time.time()

        particles_json = [
            {
                'tokens': p.context,
                'token_ids': p.context_ids,
                'weight': p.log_weight,
                'finished': p.done,
            }
            for p in particles
        ]

        gold = dev_datum.query
        db = dev_datum.schema_name

        json_result = {
            'gold': gold,
            'db_name': db,
            'question': dev_datum.utterance,
            'particles': particles_json,
            'log_ml': particles.log_ml,
            'time': end_time - start_time,
            'results': {},
        }

        def match(x, y):
            x = x.rstrip(parallel_proposal.guide.eos)
            y = y.rstrip(parallel_proposal.guide.eos)
            try:
                (exec_match, _) = evaluator.evaluate(x, y, db_name=dev_datum.schema_name)
            except Exception:
                exec_match = False
            return exec_match

        #### MBR ####

        pmax = max(
            particles,
            key=lambda candidate: particles.risk(match, ''.join(candidate.context)),
        )

        pred = ''.join(pmax.context[:-1])

        json_result['results']['mbr'] = {
            'result': evaluator.evaluate(gold, pred, db),
            'pred': pred,
            'finished': pmax.done,
            'tokens': pmax.context,
            'token_ids': pmax.context_ids,
        }

        #### Viterbi ####

        pmax = particles.particles[0]
        for p in particles.particles[1:]:
            if p.done and p.log_weight > pmax.log_weight:
                pmax = p

        pred = ''.join(pmax.context[:-1])

        print('MBR pred:', pred)

        json_result['results']['viterbi'] = {
            'result': evaluator.evaluate(gold, pred, db),
            'pred': pred,
            'finished': pmax.done,
            'tokens': pmax.context,
            'token_ids': pmax.context_ids,
        }

        print('MBR', json_result['results']['mbr']['result'])
        print('Viterbi', json_result['results']['viterbi']['result'])
        print(json.dumps(json_result), file=outfile)

        if args.decision_rule == 'mbr':
            result = json_result['results']['mbr']['result']
        else:
            assert args.decision_rule == 'viterbi'
            result = json_result['results']['viterbi']['result']

        if result[0]:
            n_correct += 1
        elif result[1] == 'invalid':
            n_invalid += 1
        elif result[1] == 'mismatch':
            n_mismatch += 1
        else:
            raise ValueError()

        n_total = sum((n_correct, n_invalid, n_mismatch))
        print(
            f'correct: {n_correct / n_total:.2f} ({n_correct}), '
            f'invalid: {n_invalid / n_total:.2f} ({n_invalid}), '
            f'mismatch: {n_mismatch / n_total:.2f} ({n_mismatch})'
        )

    print(
        json.dumps(
            {
                'correct': n_correct,
                'invalid': n_invalid,
                'mismatch': n_mismatch,
                'n_total': n_total,
            }
        ),
        file=outfile,
    )


if __name__ == '__main__':
    main()
