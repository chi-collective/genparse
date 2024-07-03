import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from ambiguous_parsing.generation.generate_pairs import generate_all_pairs

from genparse.cfglm import BoolCFGLM
from genparse.lm import AsyncGreedilyTokenizedLLM
from genparse.proposal import CharacterProposal
from genparse.vllm_compatibility import vllmpplLLM
from genparse.vllm_steer import VLLMSampler
from genparse.util import LarkStuff


def load_grammar():
    with open('amp/grammar/cfgs/lark/lark.cfg') as f:
        cfg = f.read()
    cfg = f'start: sent\n{cfg}\nignore: " "\n'
    return cfg


def load_dataset():
    dataset = []
    all_pairs = generate_all_pairs()
    for name, pairs in all_pairs.items():
        df = pd.DataFrame(pairs)
        if name.startswith('amb_'):
            assert df['surface'].unique().size * 2 == df.shape[0]
            df0 = df.loc[df.template_idx == 0, :].sort_values(by='surface')
            df1 = df.loc[df.template_idx == 1, :].sort_values(by='surface')
            assert all(df0.surface.values == df1.surface.values)
            surface = df0.surface.values
            lf0 = df0.lf.values
            lf1 = df1.lf.values
        elif name.startswith('unamb_'):
            assert df['surface'].unique().size == df.shape[0]
            df0 = df.loc[df.template_idx == 0, :].sort_values(by='surface')
            df1 = df.loc[df.template_idx == 1, :].sort_values(by='surface')
            assert (df0.size == df.size) or (df1.size == df.size)
            surface = df0.surface.values if df0.size > 0 else df1.surface.values
            lf0 = df0.lf.values if df0.size > 0 else ''
            lf1 = df1.lf.values if df1.size > 0 else ''
        else:
            raise ValueError('Unknown type')
        df = pd.DataFrame({'surface': surface, 'lf0': lf0, 'lf1': lf1, 'type': name})
        dataset.append(df)
    dataset = pd.concat(dataset).reset_index(drop=True)
    assert all(dataset.lf0.values != dataset.lf1.values)
    return dataset


def build_prompts(icl_samples, val_samples):
    prompt = (
        "Let's translate what a human user says into what a computer might say.\n\n\n"
    )
    for _, row in icl_samples.iterrows():
        prompt += f'Human: {row.surface}\nComputer: {row.lf0}\n\n'
    prompt += 'Human: %s\nComputer:'
    prompts = [prompt % row.surface for _, row in val_samples.iterrows()]
    return prompts


def build_inference_engine(model_id, batch_size, max_tokens, n_particles, random_seed):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AsyncGreedilyTokenizedLLM(
        model=vllmpplLLM(model_id), tokenizer=tokenizer, batch_size=batch_size
    )
    guide = BoolCFGLM(LarkStuff(load_grammar()).char_cfg(0.99))
    proposal = CharacterProposal(llm=llm, guide=guide)
    sampler = VLLMSampler(llm=llm, guide=guide)

    def run_inference(prompt):
        return sampler.run_inference(
            prompt=prompt,
            proposal=proposal,
            method='smc-standard',
            max_tokens=max_tokens + len(tokenizer.tokenize(prompt)),
            n_particles=n_particles,
            seed=random_seed,
            verbosity=0,
        ).posterior

    return run_inference


def evaluate_preds(pred_samples, val_samples):
    # TODO: make evaluator to assess equivalence between gold and pred examples
    # TODO: determine relevant metrics of interest
    pass


def main(args):
    dataset = load_dataset()

    amb_samples = dataset.loc[dataset.type.str.startswith('amb_'), :]
    unamb_samples = dataset.loc[dataset.type.str.startswith('unamb_'), :]

    icl_samples = unamb_samples.sample(n=args.icl_examples, random_state=args.random_seed)
    val_samples = amb_samples.sample(n=args.val_examples, random_state=args.random_seed)

    prompts = build_prompts(icl_samples, val_samples)

    run_inference = build_inference_engine(
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        n_particles=args.n_particles,
        random_seed=args.random_seed,
    )

    pred_samples = [run_inference(p) for p in tqdm(prompts)]

    evaluate_preds(pred_samples, val_samples)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='codellama/CodeLlama-7b-Instruct-hf')
    parser.add_argument('--icl_examples', default=10)
    parser.add_argument('--val_examples', default=10)
    parser.add_argument('--n_particles', default=10)
    parser.add_argument('--batch_size', default=10)
    parser.add_argument('--max_tokens', default=100)
    parser.add_argument('--random_seed', default=42)
    args = parser.parse_args()
    main(args)
