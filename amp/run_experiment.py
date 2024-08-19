import os
import vllm
import json
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from genparse.util import lark_guide
from transformers import AutoTokenizer
from genparse.lm import VirtualTokenizedLLM
from ambiguous_parsing.eval.utils import rerender
from genparse.experimental.batch_inference import smc
from genparse.experimental.batch_inference import BatchVLLM
from genparse.experimental.batch_inference import BatchStepModel
from genparse.experimental.batch_inference import ParallelCharacterProposal
from genparse.experimental.batch_inference.steer import ParticleApproximation

os.environ['TOKENIZERS_PARALLELISM'] = '(true|false)'


def sample_few_shot_prompt(
    tokenizer, dataset, query, ambiguity_type, lf0_proportion, n_shots
):
    """Samples a prompt for `ambiguity type` with lf0_proportion * n_shots of LF0 examples in expectation."""
    instances = dataset[
        (dataset['type'] == ambiguity_type) & (dataset['surface'] != query)
    ].sample(n=n_shots, random_state=seed)

    _num_lf0 = n_shots * lf0_proportion
    num_lf0 = int(
        np.ceil(_num_lf0) if random.random() < (_num_lf0 % 1) else np.floor(_num_lf0)
    )
    num_lf1 = n_shots - num_lf0

    lf0_examples = instances.head(num_lf0)[['surface', 'lf0']].to_numpy()
    lf1_examples = instances.tail(num_lf1)[['surface', 'lf1']].to_numpy()

    examples = np.concatenate((lf0_examples, lf1_examples))

    np.random.shuffle(examples)

    messages = [
        {
            'role': 'system',
            'content': 'You are a chatbot whose job is to translate what a human user says into what a computer might say.',
        },
    ]

    for surface, lf in examples:
        messages.append({'role': 'user', 'content': surface})
        messages.append({'role': 'assistant', 'content': lf})

    messages.append({'role': 'user', 'content': query})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def sample_few_shot_prompts(
    tokenizer, dataset, queries, ambiguity_types, lf0_proportion, n_shots
):
    assert len(queries) == len(ambiguity_types)
    return [
        sample_few_shot_prompt(
            tokenizer, dataset, query, ambiguity_type, lf0_proportion, n_shots
        )
        for query, ambiguity_type in zip(queries, ambiguity_types)
    ]


def sample_experiment(tokenizer, dataset, n_examples_per_type, lf0_proportion, n_shots):
    test_cases = []
    for amb_type in dataset['type'].unique():
        if not amb_type.startswith('amb'):
            continue
        test_cases.extend(
            dataset[dataset['type'] == amb_type]
            .sample(n=n_examples_per_type, random_state=seed)
            .to_dict(orient='records')
        )

    queries = [c['surface'] for c in test_cases]
    amb_tps = [c['type'] for c in test_cases]
    gt_lf0s = [c['lf0'] for c in test_cases]
    gt_lf1s = [c['lf1'] for c in test_cases]

    prompts = sample_few_shot_prompts(
        tokenizer=tokenizer,
        dataset=dataset,
        queries=queries,
        ambiguity_types=amb_tps,
        lf0_proportion=lf0_proportion,
        n_shots=n_shots,
    )

    experiment_data = {
        'prompts': prompts,
        'queries': queries,
        'amb_types': amb_tps,
        'gt_lf0s': gt_lf0s,
        'gt_lf1s': gt_lf1s,
    }

    return experiment_data


def sample_experiments(
    tokenizer, dataset, n_examples_per_type_list, lf0_proportion_list, n_shots_list
):
    experiments = []
    for n_examples_per_type in n_examples_per_type_list:
        for lf0_proportion in lf0_proportion_list:
            for n_shots in n_shots_list:
                experiment_data = sample_experiment(
                    tokenizer, dataset, n_examples_per_type, lf0_proportion, n_shots
                )
                experiments.append(
                    {
                        'n_examples_per_type': n_examples_per_type,
                        'lf0_proportion': lf0_proportion,
                        'n_shots': n_shots,
                        'data': experiment_data,
                    }
                )

    return experiments


def evaluate_posterior(posterior_approx, gt_lf0, gt_lf1):
    results = {
        'lf0_prob': 0,
        'num_lf0': 0,
        'lf1_prob': 0,
        'num_lf1': 0,
        'prob_invalid': 0,
        'num_invalid': 0,
        'gt_lf0': gt_lf0,
        'gt_lf1': gt_lf1,
        'invalid': [],
    }

    gt_lf0_canonical = rerender(gt_lf0, is_fol=True)
    gt_lf1_canonical = rerender(gt_lf1, is_fol=True)

    for lf, p in posterior_approx.items():
        lf = lf.rstrip('▪')
        try:
            canonical_pred = rerender(lf, is_fol=True)
        except Exception:
            results['prob_invalid'] += p
            results['num_invalid'] += 1
            results['invalid'].append(lf)
            continue

        if canonical_pred == gt_lf0_canonical:
            results['lf0_prob'] += p
            results['num_lf0'] += 1

        if canonical_pred == gt_lf1_canonical:
            results['lf1_prob'] += p
            results['num_lf1'] += 1

    return results


def evaluate_posteriors(posterior_approxs, gt_lf0s, gt_lf1s, types, queries):
    assert len(posterior_approxs) == len(gt_lf0s) == len(gt_lf1s)
    assert len(gt_lf1s) == len(types) == len(queries)

    items = zip(posterior_approxs, gt_lf0s, gt_lf1s, types, queries)

    all_results = []
    for posterior_approx, gt_lf0, gt_lf1, amb_type, query in items:
        results = evaluate_posterior(
            posterior_approx=posterior_approx, gt_lf0=gt_lf0, gt_lf1=gt_lf1
        )
        results['amb_type'] = amb_type
        results['query'] = query
        all_results.append(results)

    return all_results


def make_baseline_engine(llm):
    def run_baseline_experiment(
        n_particles, prompts, queries, amb_tps, gt_lf0s, gt_lf1s, out_file, **kwargs
    ):
        sampling_params = vllm.SamplingParams(
            n=n_particles, temperature=1.0, max_tokens=100, seed=0
        )

        llm_outputs = llm.generate(prompts, sampling_params)

        posterior_approxs = []
        for i, prompt_output in enumerate(llm_outputs):
            posterior = {}
            n_outputs = len(prompt_output.outputs)
            for out in prompt_output.outputs:
                pred = out.text.lstrip(' ')
                if pred in posterior:
                    posterior[pred] += 1 / n_outputs
                else:
                    posterior[pred] = 1 / n_outputs
            posterior_approxs.append(posterior)

            print(
                json.dumps(
                    {
                        'posterior': posterior,
                        'query': queries[i],
                        'prompt': prompts[i],
                        'amb_type': amb_tps[i],
                        'gt_lf0': gt_lf0s[i],
                        'gt_lf1': gt_lf1s[i],
                    }
                ),
                file=out_file,
            )

        return evaluate_posteriors(
            posterior_approxs=posterior_approxs,
            gt_lf0s=gt_lf0s,
            gt_lf1s=gt_lf1s,
            types=amb_tps,
            queries=queries,
        )

    return run_baseline_experiment


def make_genparse_engine(step_model, ess_threshold, uniform_weights=False):
    print('\t ess_threshold: ', ess_threshold)
    print('\t uniform_weights: ', uniform_weights)

    def run_genparse_experiment(
        n_particles, prompts, queries, amb_tps, gt_lf0s, gt_lf1s, out_file
    ):
        posterior_approxs = []
        for i, prompt in tqdm(enumerate(prompts)):
            step_model.set_prompt(prompt)
            posterior = smc(
                step_model,
                n_particles=n_particles,
                ess_threshold=ess_threshold,
                return_record=True,
            )

            if uniform_weights:
                posterior = ParticleApproximation(
                    particles=[p._replace(log_weight=0) for p in posterior.particles]
                )

            posterior_approxs.append(posterior.finalize(eos='▪').posterior)

            out = {
                'record': posterior.record,
                'query': queries[i],
                'prompt': prompt,
                'amb_type': amb_tps[i],
                'gt_lf0': gt_lf0s[i],
                'gt_lf1': gt_lf1s[i],
            }
            print(json.dumps(out), file=out_file)

        return evaluate_posteriors(
            posterior_approxs=posterior_approxs,
            gt_lf0s=gt_lf0s,
            gt_lf1s=gt_lf1s,
            types=amb_tps,
            queries=queries,
        )

    return run_genparse_experiment


def make_guide():
    grammar = """
    start: start_sent

    start_sent: quant " " var " . " (quant " " var " . ")* sent
    quant: "exists" | "forall"
    sent: "( " sent " )"
    | sent " " conn " " sent
    | expr "(" var ")"
    | expr "(" var ", " var ")"
    | expr "(" var ", " const ")"

    conn: "AND" | "OR"

    var: "x" | "y" | "z" | "a" | "e" | "i"

    expr: "boy" | "girl" | "man" | "woman" | "bird" | "cat" | "dog" | "fish" | "cow" | "elephant" | "book" | "rock" | "table" | "cup" | "crayon" | "telescope" | "binoculars" | "camera" | "spyglass" | "gloves" | "mittens" | "ovenmitts" | "pyjamas" | "pants" | "sweater" | "hat" | "pyjamas" | "pants" | "binoculars" | "mittens" | "ovenmitts" | "gloves" | "saw" | "observed" | "spotted" | "spied" | "picked_up" | "grabbed" | "held" | "lifted" | "heard" | "listened" | "chased" | "followed" | "called" | "ate" | "drank" | "slept" | "walked" | "left" | "played" | "moved" | "drew" | "napped" | "waved" | "smiled" | "lept" | "frowned" | "shouted" | "agent" | "patient" | "instrument" | "have"
    const: "Galileo" | "Marie" | "Sherlock" | "Ada" | "Alan" | "Katherine" | "Watson" | "Adele" | "Bill" | "Mary"
    """

    return lark_guide(grammar)


def make_genparse_model(llm, n_processes=10, max_num_particles=250, max_tokens=100):
    print('\t n_processes: ', n_processes)
    print('\t max_tokens: ', max_tokens)

    guide = make_guide()

    batch_llm = BatchVLLM(VirtualTokenizedLLM(llm.llm_engine))

    parallel_proposal = ParallelCharacterProposal(
        llm=batch_llm.llm,
        guide=guide,
        num_processes=n_processes,
        max_n_particles=max_num_particles,
        seed=seed,
    )

    step_model = BatchStepModel(
        batch_proposal=parallel_proposal, batch_llm=batch_llm, max_tokens=max_tokens
    )

    return step_model


def inference_loop(run, n_particles_list, experiments, method, out_dir):
    print(f'Running {method}...')
    out_file = open(os.path.join(out_dir, f'{method}.jsonl'), 'w')
    results = pd.DataFrame()
    for n_particles in n_particles_list:
        print('\t n_particles: ', n_particles)
        for experiment in experiments:
            print('\t lf0_proportion: ', experiment['lf0_proportion'])
            print('\t n_shots: ', experiment['n_shots'])
            result = pd.DataFrame(
                run(
                    n_particles=n_particles,
                    prompts=experiment['data']['prompts'],
                    queries=experiment['data']['queries'],
                    amb_tps=experiment['data']['amb_types'],
                    gt_lf0s=experiment['data']['gt_lf0s'],
                    gt_lf1s=experiment['data']['gt_lf1s'],
                    out_file=out_file,
                )
            )
            result['method'] = method
            result['n_samples'] = n_particles
            result['n_examples_per_type'] = experiment['n_examples_per_type']
            result['lf0_proportion'] = experiment['lf0_proportion']
            result['n_shots'] = experiment['n_shots']

            results = pd.concat([results, result], ignore_index=True)
            results.to_csv(os.path.join(out_dir, f'{method}.csv'), index=False)

    out_file.close()
    print(f'Finished {method}')


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--baseline', action='store_true')
    arg_parser.add_argument('--genparse', action='store_true')
    arg_parser.add_argument('--local', action='store_true')
    arg_parser.add_argument('--seed', type=int, default=0)
    arg_parser.add_argument('--out_dir', type=str, default='results')
    arg_parser.add_argument('--dataset_path', type=str, default='dataset.csv')
    arg_parser.add_argument(
        '--model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct'
    )
    arg_parser.add_argument('--n_particles_list', type=str, default='5,10,50')
    arg_parser.add_argument('--n_shots_list', type=str, default='10')
    arg_parser.add_argument('--lf0_prop_list', type=str, default='0,0.2,0.5,0.7,1')
    arg_parser.add_argument('--n_examples_per_type', type=int, default=50)
    arg_parser.add_argument('--ess_threshold', type=float, default=0.5)
    arg_parser.add_argument('--n_processes', type=int, default=10)
    arg_parser.add_argument('--max_tokens', type=int, default=100)

    args = arg_parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    global seed
    seed = args.seed
    print('Setting seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)

    n_particles_list = [int(n) for n in args.n_particles_list.split(',')]
    n_shots_list = [int(n) for n in args.n_shots_list.split(',')]
    lf0_proportion_list = [float(n) for n in args.lf0_prop_list.split(',')]

    print('Reading dataset...')
    dataset = pd.read_csv(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print('Sampling experiments...')
    print('\t n_examples_per_type: ', args.n_examples_per_type)
    print('\t lf0_proportion_list: ', lf0_proportion_list)
    print('\t n_shots_list: ', n_shots_list)

    experiments = sample_experiments(
        tokenizer=tokenizer,
        dataset=dataset,
        n_examples_per_type_list=[args.n_examples_per_type],
        lf0_proportion_list=lf0_proportion_list,
        n_shots_list=n_shots_list,
    )

    with open(os.path.join(args.out_dir, 'experiments.pkl'), 'wb') as f:
        pickle.dump(experiments, f)

    print('Initializing VLLM...')
    print('\t model_name: ', args.model_name)
    llm = vllm.LLM(
        model=args.model_name,
        rope_scaling={'type': 'dynamic', 'factor': 8.0},
        max_model_len=7760,
    )

    if args.baseline:
        print('Making baseline engine...')
        run_baseline_experiment = make_baseline_engine(llm)
        inference_loop(
            run=run_baseline_experiment,
            n_particles_list=n_particles_list,
            experiments=experiments,
            method='baseline',
            out_dir=args.out_dir,
        )

    if args.genparse or args.local:
        print('Initializing genparse...')
        step_model = make_genparse_model(
            llm, n_processes=args.n_processes, max_tokens=args.max_tokens
        )

    if args.genparse:
        print('Making genparse engine...')
        run_genparse_experiment = make_genparse_engine(
            step_model=step_model, ess_threshold=args.ess_threshold, uniform_weights=False
        )
        inference_loop(
            run=run_genparse_experiment,
            n_particles_list=n_particles_list,
            experiments=experiments,
            method='genparse',
            out_dir=args.out_dir,
        )

    if args.local:
        print('Making local engine...')
        run_localpoe_experiment = make_genparse_engine(
            step_model=step_model, ess_threshold=0, uniform_weights=True
        )
        inference_loop(
            run=run_localpoe_experiment,
            n_particles_list=n_particles_list,
            experiments=experiments,
            method='local',
            out_dir=args.out_dir,
        )


if __name__ == '__main__':
    main()
