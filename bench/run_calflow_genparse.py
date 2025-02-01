#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate SMCalFlow on vLLM Llama-3 (baseline).

Example usage:
python bench/run_calflow_genparse.py --train-file train_low_0.jsonl --dev-file dev_low.jsonl
"""

import argparse
import json
import logging
import os
import pprint
from pathlib import Path

from tqdm import tqdm

from bench.calflow import is_correct
from bench.calflow.datum import data_from_filename, transform_datum
from bench.calflow.bm25_index import BM25Retriever
from bench.calflow.fewshot import PromptBuilder
from genparse.util import set_seed, lark_guide_fast
from genparse.experimental.batch_inference import (
    BatchVLLM,
    ParallelCharacterProposal,
    BatchStepModel,
    smc,
    importance_sampling,
)

os.environ['TOKENIZERS_PARALLELISM'] = '(true | false)'
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-query', type=int, default=100)
    parser.add_argument(
        '--model-name',
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        # the default prompt is not intended for instruct models
    )
    parser.add_argument(
        '--method', type=str, choices=['sampling', 'greedy'], default='sampling'
    )
    parser.add_argument('--out-dir', type=str, default='runs/calflow')
    parser.add_argument('--exp-name', type=str, default='llama3-8b')
    parser.add_argument('--particles', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=300)
    parser.add_argument('--k-shot', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-processes', type=int, default=10)
    parser.add_argument('--local-poe', action='store_true')
    parser.add_argument('--return-record', action='store_true')
    parser.add_argument('--ess-threshold', type=float, default=0.5)
    parser.add_argument(
        '--train-file',
        type=str,
        default='train_low_0.jsonl',
        help='train split file name from bench clamp',
    )
    parser.add_argument(
        '--dev-file',
        type=str,
        default='dev_low.jsonl',
        help='dev split file name from bench clamp',
    )

    return parser


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
    )
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    logger.info(f'Config: {pprint.pformat(vars(args))}')

    logger.info('loading data...')
    calflow_data_root = Path(
        '/home/leodu/semantic_parsing_with_constrained_lm/data/benchclamp/processed/CalFlowV2'
    )
    train_data_file = calflow_data_root / args.train_file
    dev_data_file = calflow_data_root / args.dev_file

    train_data = transform_datum(data_from_filename(train_data_file))
    dev_data = transform_datum(data_from_filename(dev_data_file))
    logger.info('finished loading data.')

    logger.info(f'Example datum:\n {pprint.pformat(train_data[0])}')

    train_retriever = BM25Retriever(
        train_data=train_data, top_k=args.k_shot, best_first=False
    )
    prompt_builder = PromptBuilder.for_demo(
        do_include_context=False, use_preamble=True
    )  # configs.lib.common:83

    access_token = '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'
    os.environ['HF_TOKEN'] = access_token

    # Prepare the prompts and collect them in `samples`
    samples = []
    for dev_datum in tqdm(dev_data):
        selected_train_data = train_retriever(dev_datum)
        prompt = prompt_builder.assemble(selected_train_data, dev_datum)
        samples.append({'datum': dev_datum, 'prompt': prompt})

    logger.debug(f'Example prompt:\n{samples[0]["prompt"]}')

    batch_llm = BatchVLLM.from_name(args.model_name)

    logger.info('loading grammar...')
    grammar_file = 'benchmark/grammars/smcalflow.lark'
    with open(grammar_file, 'r') as f:
        grammar = f.read()
    guide = lark_guide_fast(grammar)
    logger.info('grammar and parser loaded.')

    logger.info('constructing proposal...')
    n_particles = args.particles
    parallel_proposal = ParallelCharacterProposal(
        llm=batch_llm.llm,
        guide=guide,
        num_processes=args.n_processes,
        max_n_particles=100,  # should be >= the number of particles you plan to run inference with
        seed=0,
    )
    logger.info('proposal constructed.')

    step_model = BatchStepModel(
        batch_proposal=parallel_proposal, batch_llm=batch_llm, max_tokens=args.max_tokens
    )

    outpath = os.path.join(
        args.out_dir,
        f'{args.exp_name}-p{args.particles}-lpoe-{args.local_poe}-shot-{args.k_shot}.jsonl',
    )
    outfile = open(outpath, 'w+')
    logger.info(f'writing results to {outpath} ...')

    n_correct = 0
    cumulative_weighted_acc = 0
    n_total = 0
    for sample in tqdm(samples):
        n_total += 1
        prompt = sample['prompt']

        step_model.set_prompt(prompt)

        if args.local_poe:
            particles = importance_sampling(
                step_model, n_particles, return_record=args.return_record
            )
        else:
            particles = smc(
                step_model,
                n_particles=n_particles,
                return_record=args.return_record,
                ess_threshold=args.ess_threshold,
                verbosity=0,
            )

        weighted_acc = 0
        for pred, p in particles.posterior.items():
            acc = is_correct(pred.rstrip(parallel_proposal.eos), sample['datum'])
            weighted_acc += p * acc
        cumulative_weighted_acc += weighted_acc

        best = particles.particles[0]
        for p in particles.particles[1:]:
            if p.done and p.log_weight > best.log_weight:
                best = p
        pred = ''.join(best.context[:-1])

        eval_result: bool = is_correct(pred, sample['datum'])
        if eval_result:
            n_correct += 1
        logger.info(
            f' {n_correct} out of {n_total} is correct, acc={n_correct / n_total:.3f}, '
            f'total weighted acc: {cumulative_weighted_acc / n_total:.3f}, '
            f'weighted acc: {weighted_acc:.3f}'
        )

        datum = sample['datum']
        json_result = dict(
            dialogue_id=datum.dialogue_id,
            turn_part_index=datum.turn_part_index,
            natural=datum.natural,
            canonical=datum.canonical,
            pred=pred,
            record=particles.record,
            result=eval_result,
        )
        print(json.dumps(json_result), file=outfile)


if __name__ == '__main__':
    main()
