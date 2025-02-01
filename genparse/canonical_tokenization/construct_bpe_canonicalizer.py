import argparse
import json
import pathlib
from typing import Any

import torch
import transformers

from genparse.canonical_tokenization.berglund import TokenDFA


def get_tokenizer_data_from_args(args, parser):
    with get_tokenizer_json_path_from_args(args, parser).open() as fin:
        return json.load(fin)


def get_tokenizer_json_path_from_args(args, parser):
    if (args.hugging_face_name is None) == (args.json_file is None):
        parser.error('one of --hugging-face-name or --json-file is required')
    if args.hugging_face_name is not None:
        return get_json_path_from_hf_name(args.hugging_face_name)
    else:
        return args.json_file


def get_json_path_from_hf_name(name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model_path = pathlib.Path(tokenizer.vocab_file)
    return model_path.with_name(tokenizer.vocab_files_names['tokenizer_file'])


def construct_dictionary_from_tokenizer(
    tokenizer_data: dict[str, Any],
) -> tuple[list[int], list[tuple[int, int, int]]]:
    model_data = tokenizer_data['model']
    vocab_str_to_int = model_data['vocab']
    merge_rules = [x.split(' ', 1) for x in model_data['merges']]

    # TODO There are a few kinds of token that are not composite tokens but are
    # also more than one character long:
    # - <s> (BOS): Should never appear any time during decoding, so this should
    #     always be masked out.
    # - </s> (EOS): Should be allowed iff the current state is an accept state.
    # - <unk>: With byte fallback, this should be impossible and should always
    #     be masked out.
    # - <0xFF> (byte tokens): Used for byte fallback. These can only form valid
    #     UTF-8 byte sequences. Handling this will be a little complicated, but
    #     should be expressible as a token DFA and so compatible with the
    #     Berglund algorithm. However, for full correctness it's necessary to
    #     prohibit UTF-8 sequences of characters that are in the vocabulary,
    #     and this is complicated.
    #     A good approach to solving this is probably not to encode these
    #     constraints in the DFA (since it would take a large number of states
    #     and transitions), but to modify the code for simulating the DFA to
    #     enforce UTF-8 validity. This should be provably correct because none
    #     of the merge rules can involve byte tokens.
    # - ▁<EOT><EOT>, ▁<EOT><EOT><EOT>, etc.: I don't know what these are for.

    # TODO Handle special tokens more generally. Don't hard-code these values.
    del vocab_str_to_int['<s>']
    del vocab_str_to_int['</s>']

    # TODO Handle byte fallback correctly. For now, just disallow <unk> and all
    # byte tokens.
    if model_data['byte_fallback']:
        del vocab_str_to_int[model_data['unk_token']]
        for token in generate_byte_tokens():
            del vocab_str_to_int[token]

    # Free memory.
    del tokenizer_data
    del model_data

    # Get the base alphabet. It's the set of all tokens that can't be formed by
    # a merge rule.
    composite_tokens = {u + v for u, v in merge_rules}
    base_alphabet_as_int = [
        v for k, v in vocab_str_to_int.items() if k not in composite_tokens
    ]
    del composite_tokens

    # Convert the merge rules to ints.
    merge_rules_as_int = [
        (vocab_str_to_int[u], vocab_str_to_int[v], vocab_str_to_int[u + v])
        for u, v in merge_rules
    ]

    return base_alphabet_as_int, merge_rules_as_int


def construct_canonicalizer_from_tokenizer(
    tokenizer_data: dict[str, Any], output_path: pathlib.Path
) -> None:
    """Note that tokenizer_data may be modified in-place."""
    # TODO Validate tokenizer attributes (the model is BPE, etc.)
    model_data = tokenizer_data['model']
    if model_data['type'] != 'BPE':
        raise ValueError
    vocabulary_size = len(model_data['vocab'])
    # TODO Don't hard-code </s>.
    eos_token_id = model_data['vocab']['</s>']
    base_alphabet, dictionary = construct_dictionary_from_tokenizer(tokenizer_data)

    # Free memory.
    del tokenizer_data
    del model_data

    # Run Berglund's algorithm to construct the token DFA.
    dfa = TokenDFA.from_dictionary(base_alphabet, dictionary)

    output_path.mkdir(exist_ok=True)
    save_transitions(dfa.get_transitions(), output_path)
    with (output_path / 'metadata.json').open('w') as fout:
        json.dump(
            dict(
                vocabulary_size=vocabulary_size,
                eos_token_id=eos_token_id,
                num_states=dfa.num_states(),
            ),
            fout,
        )


def save_transitions(transitions, output_path):
    with (output_path / 'transitions.csv').open('w') as fout:
        for state_from, symbol, state_to in transitions:
            fout.write(f'{state_from},{symbol},{state_to}\n')


def generate_byte_tokens():
    digits = '0123456789ABCDEF'
    for d1 in digits:
        for d2 in digits:
            yield f'<0x{d1}{d2}>'


def main():
    parser = argparse.ArgumentParser(
        description='Given a pretrained Hugging Face BPE model, construct a data '
        'structure that can be used to ensure that only canonical '
        'tokenizations are used.'
    )
    parser.add_argument(
        '--hugging-face-name', help='Name of a model on Hugging Face hub to use.'
    )
    parser.add_argument(
        '--json-file',
        type=pathlib.Path,
        help='Path to a .json file defining a Hugging Face tokenizer.',
    )
    parser.add_argument(
        '--output',
        type=pathlib.Path,
        required=True,
        help='Path to an output .pt file where the canonicalizer will be written.',
    )
    args = parser.parse_args()

    construct_canonicalizer_from_tokenizer(
        get_tokenizer_data_from_args(args, parser), args.output
    )


if __name__ == '__main__':
    main()
