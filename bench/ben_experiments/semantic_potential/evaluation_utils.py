import numpy as np
from tqdm import tqdm
from genparse import EOS
import concurrent.futures
from arsenal.maths import logsumexp
from functools import lru_cache, wraps
from bench.spider.evaluator import Evaluator
from genparse.batch_inference.steer import ParticleApproximation, Particle


def get_final_particles_from_record(record):
    final_step = record['history'][-1]
    particles = final_step['particles']
    if 'resample_indices' in final_step:
        particles = [
            {**particles[i], **{'weight': final_step['average_weight']}}
            for i in final_step['resample_indices']
        ]
    return particles


def create_particles(record_particles):
    return [
        Particle(
            prompt=None,
            context=p['context'],
            context_ids=p['context_ids'],
            done=False,
            log_weight=p['weight'],
            parent=None,
        )
        for p in record_particles
    ]


def create_particle_approx(particles):
    return ParticleApproximation(
        [
            Particle(
                prompt=None,
                context=p['context'],
                context_ids=p['context_ids'],
                done=True,
                log_weight=p['weight'],
                parent=None,
            )
            for p in particles
        ]
    )


def posterior_weighted_eval(particles, gold, db):
    weighted_acc = 0
    particle_results = []

    log_weights = [p.log_weight for p in particles]
    log_total = logsumexp(log_weights)
    log_normalized_weights = log_weights - log_total
    probs = np.exp(log_normalized_weights)

    for i, particle in enumerate(particles):
        pred = ''.join(particle.context).rstrip(EOS)
        p = probs[i]
        if np.isnan(p):
            p = 0
        acc = cached_eval(gold, pred, db)
        particle_results.append((particle.context, acc))
        weighted_acc += p * acc[0]

    return {'result': weighted_acc, 'particle_results': particle_results}


def process_datum_wrapper(args):
    datum, overwrite = args
    return process_datum(datum, overwrite)


def process_datum(datum, overwrite):
    particles = get_final_particles_from_record(datum['record'])
    approx = create_particle_approx(particles)

    results = {}
    gold = datum['gold']
    db = datum['db_name']

    if overwrite or ('posterior_weighted_acc' not in datum['results']):
        results['posterior_weighted_acc'] = posterior_weighted_eval(approx, gold, db)

    datum['results'].update(results)

    return datum


def run_and_add_evaluation(
    data, n_workers, raw_spider_dir, overwrite=False, timeout=None
):
    with concurrent.futures.ProcessPoolExecutor(
        initializer=initialize_worker,
        initargs=(
            raw_spider_dir,
            timeout,
        ),
        max_workers=n_workers,
    ) as executor:
        with tqdm(total=len(data)) as progress_bar:
            results = []
            args = [(datum, overwrite) for datum in data]
            futures = [executor.submit(process_datum_wrapper, arg) for arg in args]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                progress_bar.update(1)

    return results


@lru_cache
def cached_eval(x, y, db):
    return evaluator.evaluate(x, y, db_name=db)


def initialize_worker(raw_spider_dir, timeout=None):
    global evaluator
    evaluator = Evaluator(raw_spider_dir, timeout=timeout)
