##############
# Evaluation #
##############

import os
import json
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from functools import lru_cache, wraps
from bench.spider.evaluator import Evaluator

from genparse.batch_inference.steer import ParticleApproximation, Particle

from genparse import EOS

eos = EOS
spider_dir = Path('../../../spider/data/spider')


def get_final_particles_from_record(record):
    final_step = record['history'][-1]
    particles = final_step['particles']
    if 'resample_indices' in final_step:
        particles = [
            {**particles[i], **{'weight': final_step['average_weight']}}
            for i in final_step['resample_indices']
        ]
    return particles


def create_particle_approx(particles):
    return ParticleApproximation(
        [
            Particle(
                prompt=None,
                context=p['context'],
                context_ids=p['context_ids'],
                done=True,
                log_weight=p['weight'],
                log_weight_updates=None,
                parent=None,
            )
            for p in particles
        ]
    )


def mbr_eval(particles, gold, db):
    def match(x, y):
        x = x.rstrip(eos)
        y = y.rstrip(eos)
        try:
            (exec_match, _) = cached_eval(x, y, db_name=db)
        except Exception:
            exec_match = False
        return exec_match

    pmax = max(
        particles,
        key=lambda candidate: particles.risk(match, ''.join(candidate.context)),
    )

    pred = ''.join(pmax.context[:-1])

    return {
        'result': cached_eval(gold, pred, db),
        'pred': pred,
        'finished': pmax.done,
        'tokens': pmax.context,
        'token_ids': pmax.context_ids,
    }


def viterbi_eval(particles, gold, db):
    pmax = particles.particles[0]
    for p in particles.particles[1:]:
        if p.done and p.log_weight > pmax.log_weight:
            pmax = p

    pred = ''.join(pmax.context).rstrip(eos)

    return {
        'result': cached_eval(gold, pred, db),
        'pred': pred,
        'finished': pmax.done,
        'tokens': pmax.context,
        'token_ids': pmax.context_ids,
    }


def posterior_weighted_eval(particles, gold, db):
    weighted_acc = 0
    particle_results = {}
    for pred, p in particles.posterior.items():
        if np.isnan(p):
            p = 0
        pred = pred.rstrip(eos)
        acc = cached_eval(gold, pred, db)
        assert pred not in particle_results, pred
        particle_results[pred] = acc
        weighted_acc += p * acc[0]

    return {'result': weighted_acc, 'particle_results': particle_results}


@lru_cache
def cached_eval(x, y, db):
    return evaluator.evaluate(x, y, db_name=db)


def initialize_worker(timeout=None):
    global evaluator
    evaluator = Evaluator(spider_dir, timeout=timeout)


def process_datum_wrapper(args):
    datum, run_mbr, overwrite = args
    return process_datum(datum, run_mbr, overwrite)


def process_datum(datum, run_mbr, overwrite):
    particles = get_final_particles_from_record(datum['record'])
    approx = create_particle_approx(particles)

    results = {}
    gold = datum['gold']
    db = datum['db_name']

    if overwrite or ('posterior_weighted_acc' not in datum['results']):
        results['posterior_weighted_acc'] = posterior_weighted_eval(approx, gold, db)

    if overwrite or ('mbr' not in datum['results']):
        if run_mbr:
            results['mbr'] = mbr_eval(approx, gold, db)
        else:
            results['mbr'] = None

    if overwrite or ('viterbi' not in datum['results']):
        results['viterbi'] = viterbi_eval(approx, gold, db)

    datum['results'].update(results)

    return datum


from utils import _iter_args, make_file_path, read_file


def run_evaluations(
    results_dir,
    models,
    methods,
    runs,
    ess_thresholds,
    n_particles_list,
    proposals,
    n_workers,
    run_mbr=True,
    overwrite=False,
    timeout=None,
):
    for args in _iter_args(
        models, methods, runs, ess_thresholds, n_particles_list, proposals
    ):
        model, method, run, ess, n_particles, proposal = args

        data = read_file(
            results_dir=results_dir,
            model=model,
            method=method,
            run=run,
            ess=ess,
            n_particles=n_particles,
            proposal=proposal,
        )

        if data == []:
            continue

        data = run_and_add_evaluation(
            data, n_workers, run_mbr=run_mbr, overwrite=overwrite, timeout=timeout
        )

        fp = make_file_path(
            results_dir=results_dir,
            model=model,
            method=method,
            run=run,
            ess=ess,
            n_particles=n_particles,
            proposal=proposal,
        )

        write_evaluated_results(data, fp)


def run_and_add_evaluation(data, n_workers, run_mbr=True, overwrite=False, timeout=None):
    with concurrent.futures.ProcessPoolExecutor(
        initializer=initialize_worker, initargs=(timeout,), max_workers=n_workers
    ) as executor:
        with tqdm(total=len(data)) as progress_bar:
            results = []
            args = [(datum, run_mbr, overwrite) for datum in data]
            futures = [executor.submit(process_datum_wrapper, arg) for arg in args]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                progress_bar.update(1)

    return results


def write_evaluated_results(data, fp):
    with open(fp, 'w') as f:
        for l in data:
            print(json.dumps(l), file=f)
