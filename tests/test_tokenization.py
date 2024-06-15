from transformers import AutoTokenizer

from genparse.tokenization import decode_tokenizer_vocab

CASES = [
    'state',
    ' state',
    ' SELECT state',
    '\n',
    '\t',
    '.',
    ':',
    '+=',
    'SELECT * FROM AIRPORTS WHERE city  =  "Anthony"	flight_2',
    ' SELECT count(*)  FROM FLIGHTS  AS T1 JOIN AIRPORTS AS T2  ON T1.DestAirport',
    '  SELECT Name   FROM country WHERE IndepYear  >  1950	world_1  ',
]


def test_codellama():
    tokenizer = AutoTokenizer.from_pretrained(
        'codellama/CodeLlama-7b-Instruct-hf',
        use_fast=True,
        prefix_token=None,
        middle_token=None,
        suffix_token=None,
        eot_token=None,
        fill_token=None,
    )

    decoded = decode_tokenizer_vocab(tokenizer)

    decoded_size = len(decoded)
    tok_size = tokenizer.vocab_size
    assert decoded_size == tok_size, [decoded_size, tok_size]

    for case in CASES:
        encd = tokenizer.encode(case)
        have = ''.join([decoded[i] for i in encd])
        want = tokenizer.decode(encd)
        assert want == have, [want, have, case]

        # NOTE: codellama's `encode` prepends a space to everything
        # there doesn't seem to be any way to disable this behaviour


def test_gpt2():
    tokenizer = AutoTokenizer.from_pretrained(
        'gpt2',
        use_fast=True,
        prefix_token=None,
        middle_token=None,
        suffix_token=None,
        eot_token=None,
        fill_token=None,
    )

    decoded = decode_tokenizer_vocab(tokenizer)

    decoded_size = len(decoded)
    tok_size = tokenizer.vocab_size
    assert decoded_size == tok_size, [decoded_size, tok_size]

    for case in CASES:
        encd = tokenizer.encode(case)
        have = ''.join([decoded[i] for i in encd])
        want = tokenizer.decode(encd)
        assert want == have, [want, have, case]


def test_t5():
    tokenizer = AutoTokenizer.from_pretrained(
        'google-t5/t5-small',
        use_fast=True,
        prefix_token=None,
        middle_token=None,
        suffix_token=None,
        eot_token=None,
        fill_token=None,
    )

    decoded = decode_tokenizer_vocab(tokenizer)

    decoded_size = len(decoded)
    tok_size = tokenizer.vocab_size
    assert decoded_size == tok_size, [decoded_size, tok_size]

    # tokenizer.add_eos_token = False # NOTE: disabling not possible for this tokenizer
    # NOTE: T5 tokenizer removes consecutive occurences of whitespace tokens
    # and always adds a space to the begining of the string, unless there already is one
    # or the string is empty. smh.

    for case in CASES:
        encd = tokenizer.encode(case)
        have = ''.join([decoded[i] for i in encd])
        want = tokenizer.decode(encd)
        have = have[1:] if have.startswith(' ') else have
        assert want == have, [want, have, case]


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
