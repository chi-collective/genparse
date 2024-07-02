import json
import timeit

import memory_profiler

from genparse.canonical_tokenization.berglund import TokenDFA
from genparse.canonical_tokenization.construct_bpe_canonicalizer import (
    get_json_path_from_hf_name,
    construct_dictionary_from_tokenizer,
)


def read_json_file(path):
    with path.open() as fin:
        return json.load(fin)


def main():
    model = 'codellama/CodeLlama-7b-Instruct-hf'
    merge_rule_limit = 1000
    num_repetitions = 10
    base_alphabet, dictionary = construct_dictionary_from_tokenizer(
        read_json_file(get_json_path_from_hf_name(model))
    )
    dictionary = dictionary[:merge_rule_limit]

    total_time = None

    def run_timeit():
        nonlocal total_time
        total_time = timeit.timeit(
            lambda: TokenDFA.from_dictionary(base_alphabet, dictionary),
            number=num_repetitions,
            setup='gc.enable()',
        )

    print('starting...')
    memory = memory_profiler.memory_usage((run_timeit,))
    max_memory = max(memory)
    mean_time = total_time / num_repetitions
    print(f'memory: {max_memory:.2f} MB')
    print(f'time:   {mean_time:.3f} s')


if __name__ == '__main__':
    main()
