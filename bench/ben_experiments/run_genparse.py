#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import psutil
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp

from bench.cache import ProposalCache
from bench.spider.evaluator import Evaluator
from utils import (
    load_spider_data,
    load_prompt_formatter,
    reformat_grammar,
    HF_ACCESS_TOKEN,
)

from genparse.util import set_seed
from genparse.experimental.batch_inference import (
    BatchVLLM,
    BatchStepModel,
    smc,
    importance_sampling,
)
from genparse.experimental.batch_inference.steer import ParticleApproximation, Particle

os.environ['TOKENIZERS_PARALLELISM'] = '(true | false)'

logger = logging.getLogger(__name__)


def int_or_none(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f'Invalid integer value: {value}')


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
        type=int_or_none,
        default=0,
        help='parameter for token proposal distribution',
    )
    parser.add_argument('--n-processes', type=int_or_none, default=None)
    parser.add_argument('--out-dir', default='', type=str)
    parser.add_argument(
        '--schema', type=str, help='Schema to evaluate, seperated by comma', default='all'
    )
    parser.add_argument(
        '--improper-weights',
        action='store_true',
        help='IS + improper importance weights when used',
    )
    parser.add_argument('--verbosity', type=int, default=0)

    return parser


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


def make_example_key(schema_name, question):
    return (schema_name, question)


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )

    parser = get_argparser()
    args = parser.parse_args()

    if args.improper_weights:
        print('Evaluating with improper weights')

    outpath = os.path.join(
        args.out_dir, f'{args.exp_name}-p{args.particles}-{args.proposal}'
    )

    if args.K != 0:
        outpath += f'-K{args.K}'

    json.dump(vars(args), open(f'{outpath}-args.json', 'w'), indent=4)

    outpath += '.jsonl'

    already_processed = []
    if os.path.exists(outpath):
        with open(outpath, 'r') as lines:
            for line in lines:
                line = json.loads(line)
                already_processed.append(
                    make_example_key(line['db_name'], line['question'])
                )
        outfile = open(outpath, 'a+')
        print(f'Found {len(already_processed)} already done at {outpath}')
    else:
        outfile = open(outpath, 'w+')

    set_seed(args.seed)

    raw_spider_dir = Path('../spider/data/spider')
    spider_dev_data = load_spider_data(raw_spider_dir, split='dev')
    evaluator = Evaluator(raw_spider_dir)
    prompt_formatter = load_prompt_formatter(raw_spider_dir)

    if len(already_processed) >= len(spider_dev_data):
        return

    with open('../../benchmark/grammars/spider_schema_grammar.json', 'r') as f:
        all_grammars = json.load(f)

    if args.schema == 'all':
        schema = list(set([d.schema_name for d in spider_dev_data]))
        print(f'Using schema {",".join(schema)}')
    else:
        schema = args.schema.split(',')

    proposal_cache = ProposalCache(guide_cache_path='guide_cache.pkl', maxsize=1)

    print(f'Initialized {proposal_cache}')

    if isinstance(args.K, int):
        proposal_args = {'K': args.K} if args.K > 0 else {}
    elif args.K is None:
        proposal_args = {'K': args.K}
    else:
        raise ValueError(f'Invalid K {args.K}')

    if args.n_processes is None:
        n_processes = min(args.particles, mp.cpu_count() - 2)
    elif isinstance(args.n_processes, int):
        n_processes = args.n_processes
    else:
        raise ValueError(f'Invalid n_processes {args.n_processes}')

    os.environ['HF_TOKEN'] = HF_ACCESS_TOKEN
    batch_llm = BatchVLLM.from_name(args.model_name)
    tokenizer = batch_llm.get_tokenizer()

    n_correct, n_invalid, n_mismatch = 0, 0, 0

    skipped_schema = []

    for i, dev_datum in tqdm(
        enumerate(spider_dev_data), total=len(spider_dev_data), smoothing=0.0
    ):
        if dev_datum.schema_name not in schema:
            if dev_datum.schema_name not in skipped_schema:
                skipped_schema.append(dev_datum.schema_name)
                print(f'Skipping schema {dev_datum.schema_name}')
            continue

        if (
            make_example_key(dev_datum.schema_name, dev_datum.utterance)
            in already_processed
        ):
            print(f'Skipping {i}')
            continue

        messages = prompt_formatter.format_openai(dev_datum)

        grammar = reformat_grammar(all_grammars[dev_datum.schema_name])

        current_mem_usage = psutil.virtual_memory().percent
        if current_mem_usage > 80:
            proposal_cache.evict_objects()
            print(
                'Evicted proposals from cache:'
                f' prev_mem_usage={current_mem_usage=}% -->'
                f' curr_mem_usage={psutil.virtual_memory().percent}%'
            )

        parallel_proposal = proposal_cache.fetch_or_create_proposal(
            llm=batch_llm.llm,
            grammar=grammar,
            proposal_name=args.proposal,
            n_processes=n_processes,
            proposal_args=proposal_args,
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

        if args.improper_weights:
            particles = importance_sampling(step_model, n_particles=args.particles)
            particles = ParticleApproximation(
                [
                    Particle(
                        prompt=None,
                        context=p.context,
                        context_ids=p.context_ids,
                        done=p.done,
                        log_weight=0,
                        parent=None,
                    )
                    for p in particles.particles
                ]
            )
        else:
            particles = smc(step_model, n_particles=args.particles)

        end_time = time.time()

        del step_model

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

        json_result['results']['mbr'] = mbr_eval(
            particles, evaluator, gold, db, parallel_proposal.eos
        )

        json_result['results']['posterior_weighted_acc'] = posterior_weighted_eval(
            particles, evaluator, gold, db, parallel_proposal.eos
        )

        json_result['results']['viterbi'] = (
            viterbi_eval(particles, evaluator, gold, db, parallel_proposal.eos)
            if not args.improper_weights
            else None
        )

        print(json.dumps(json_result), file=outfile)

        if args.verbosity > 0:
            print(f"MBR: {json_result['results']['mbr']['pred']}")
            print(f"Viterbi: {json_result['results']['viterbi']['pred']}")

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


if __name__ == '__main__':
    main()
