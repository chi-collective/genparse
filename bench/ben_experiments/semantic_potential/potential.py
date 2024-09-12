import os
import pickle
import numpy as np
from genparse import EOS
from functools import partial

# Example usage
# prompt_formatter = partial(utterance_prompter, spider_train_data=train_data)
# scorer = CachedScorer(batch_llm)
# potential = partial(
#    utterance_potential,
#    scorer=scorer,
#    prompt_formatter=prompt_formatter,
#    temperature_schedule=lambda t : 1
# )


def utterance_prompter(contexts, are_done, tokenizer, spider_train_data):
    system_prompt = (
        'You are a coding assistant helping an analyst understand the intent behind SQL queries. '
        'More specifically, the analyst provides you with a *partial* SQL query, and in response, '
        'you must guess a possible english language question which an SQL statement with that begining could be answering. '
        'There are many correct answers! Your goal is to infer a possible intent behind the SQL query without having observed the full query.'
    )

    user_message_template = (
        '\nPlease write a question that the SQL query might be answering: {query}'
    )

    example_ids = [10, 100, 1000, 3000, 4000, 5000, 6000, 6500]

    examples, utterances = [], []
    for example_id in example_ids:
        example = spider_train_data[example_id]
        examples.append(tokenizer.encode(example.query, add_special_tokens=False))
        utterances.append(example.utterance)

    prompts = []
    for i, context in enumerate(contexts):
        messages = [{'role': 'system', 'content': system_prompt}]
        step_num = len(context) if not are_done[i] else None
        for utterance, example in zip(utterances, examples):
            trunc_example = tokenizer.decode(
                example[:step_num] if isinstance(step_num, int) else example
            )
            messages.append(
                {
                    'role': 'user',
                    'content': user_message_template.format(query=trunc_example),
                }
            )
            messages.append({'role': 'system', 'content': utterance})
        messages.append(
            {
                'role': 'user',
                'content': user_message_template.format(
                    query=''.join(context).rstrip(EOS)
                ),
            }
        )
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    return prompts


class CachedScorer:
    def __init__(self, batch_llm, cache_path='scorer_cache.pkl'):
        self.batch_llm = batch_llm
        self.cache_path = cache_path
        self.cache = self.load_cache()

    def __call__(self, prompts, token_ids, temperature):
        prompt2score = {}
        new_prompts = []
        for prompt in prompts:
            key = (prompt, tuple(token_ids), temperature)
            if key in self.cache:
                prompt2score[prompt] = self.cache[key]
            else:
                new_prompts.append(prompt)

        if new_prompts:
            scores = self.batch_llm.batch_score_sequences(
                prompts=new_prompts, token_ids=token_ids, temperature=temperature
            )
            for prompt, score in zip(new_prompts, scores):
                prompt2score[prompt] = score
                self.cache[(prompt, tuple(token_ids), temperature)] = score

        return [prompt2score[prompt] for prompt in prompts]

    def clear_cache(self):
        self.cache = {}

    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def save_cache(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)


def utterance_potential(
    step_num,
    particles,
    scorer,
    utterance_ids,
    prompt_formatter,
    temperature_schedule,
    normalize_strings=False,
):
    temperature = temperature_schedule(step_num)

    if temperature == np.inf:
        return [0] * len(particles)

    are_done = []
    unique_contexts = []
    particle_idx_to_prompt_idx = []
    for idx, particle in enumerate(particles):
        if not normalize_strings:
            context = particle.context
        else:
            context = [c.lower() for c in particle.context]

        if context not in unique_contexts:
            unique_contexts.append(context)
            are_done.append(particle.done)
            particle_idx_to_prompt_idx.append(len(unique_contexts) - 1)
        else:
            particle_idx_to_prompt_idx.append(unique_contexts.index(context))

    prompts = prompt_formatter(unique_contexts, are_done)

    logprobs = scorer(prompts=prompts, token_ids=utterance_ids, temperature=temperature)

    potential_values = [
        logprobs[particle_idx_to_prompt_idx[p_idx]] for p_idx in range(len(particles))
    ]

    return potential_values
