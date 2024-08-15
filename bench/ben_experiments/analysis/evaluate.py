##############
# Evaluation #
##############

import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from functools import lru_cache
from bench.spider.evaluator import Evaluator
from genparse.experimental.batch_inference.steer import ParticleApproximation, Particle

from genparse import EOS

EOS
eos = EOS
spider_dir = Path('../../spider/data/spider')


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


def spider_eval(particles, gold, db, run_mbr):
    approx = create_particle_approx(particles)
    return {
        'posterior_weighted_acc': posterior_weighted_eval(approx, gold, db),
        'mbr': mbr_eval(approx, gold, db) if run_mbr else None,
        'viterbi': viterbi_eval(approx, gold, db),
    }


def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    print(f'Function {func.__name__} timed out after {seconds} seconds')
                    return (False, 'invalid')

        return wrapper

    return decorator


@lru_cache
@timeout(10)
def cached_eval(x, y, db):
    return evaluator.evaluate(x, y, db_name=db)


def initialize_worker():
    global evaluator
    evaluator = Evaluator(spider_dir)


def process_datum_wrapper(args):
    datum, run_mbr, overwrite = args
    return process_datum(datum, run_mbr, overwrite)


def process_datum(datum, run_mbr, overwrite):
    particles = get_final_particles_from_record(datum['record'])
    if not overwrite and len(datum['results']) > 0:
        return datum
    datum['results'].update(
        spider_eval(particles, gold=datum['gold'], db=datum['db_name'], run_mbr=run_mbr)
    )
    return datum


def run_and_add_evaluation(data, n_workers, run_mbr=True, overwrite=False):
    with concurrent.futures.ProcessPoolExecutor(
        initializer=initialize_worker, max_workers=n_workers
    ) as executor:
        with tqdm(total=len(data)) as progress_bar:
            results = []
            args = [(datum, run_mbr, overwrite) for datum in data]
            for result in executor.map(process_datum_wrapper, args):
                results.append(result)
                progress_bar.update(1)

    return results
