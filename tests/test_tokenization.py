from transformers import AutoTokenizer

from genparse.tokenization import decode_tokenizer_vocab
from genparse.lm import TokenizedLLM


cases = [
    'state',
    ' state',
    ' SELECT state',
    '\n',
    '\t',
    '.',
    'SELECT * FROM AIRPORTS WHERE city  =  "Anthony"	flight_2',
]

failed_cases = [  # just a few of the likely many failure modes with the current method
    '\t\n\r',
    '®',
    '’•¶∂ƒ˙∆£Ħ爨ൠᅘ∰፨',
    '×',
    '¯',
    '¬',
]

tokenizer_names = [
    'codellama/CodeLlama-7b-Instruct-hf',
    # "meta-llama/Meta-Llama-3-8B",
    'gpt2',
]

tokenizers = [
    (tokenizer_name, AutoTokenizer.from_pretrained(tokenizer_name))
    for tokenizer_name in tokenizer_names
]


class DummyLM:
    pass


def test_decoding():
    for tokenizer_name, tokenizer in tokenizers:
        decoded = decode_tokenizer_vocab(tokenizer)

        decoded_size = len(decoded)
        tok_voc_size = len(tokenizer)
        assert decoded_size == tok_voc_size, [decoded_size, tok_voc_size, tokenizer_name]

        for case in cases:
            encd = tokenizer.encode(case)
            have = ''.join([decoded[i] for i in encd])
            want = tokenizer.decode(encd)
            assert want == have, [want, have, case, tokenizer_name]


def test_encoding():
    for tokenizer_name, tokenizer in tokenizers:
        llm = TokenizedLLM(model=DummyLM(), tokenizer=tokenizer, batch_size=0)

        for case in cases:
            want = llm.tokenizer.encode(case)
            have = [llm._encode[i] for i in llm.encode_prompt(case)]
            assert want == have, [want, have, tokenizer_name]


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
