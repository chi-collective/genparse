"""
Tools for tokenization.
"""

from arsenal import defaultdict


# def ints2bytes(sequence):
#    # check in the range of 0-255
#    for item in sequence:
#        if not 0 <= item <= 255:
#           raise ValueError(f'item: {item} is not in the range [0, 255]')
#    return bytes(sequence)


# def bytes2ints(byte_sequence):
#    return list(byte_sequence)


def get_tokenizer_mapping(tokenizer):
    name = tokenizer.name_or_path.lower()
    if 'gpt2' in name:
        return GPT2Mapping(tokenizer)
    elif 'codellama' in name:
        return CodeLLaMaMapping(tokenizer)
    elif 'llama-3' in name:
        return LLaMaMapping(tokenizer)
    else:
        raise ValueError(
            f'Unknown tokenizer type: {tokenizer.name_or_path}.'
            f'GenParse supports the following tokenizers: gpt2, codellama, llama-3'
        )


def decode_tokenizer_vocab(tokenizer):
    mapping = get_tokenizer_mapping(tokenizer)
    decoded = mapping.post_process([mapping(i) for i in range(len(tokenizer))])

    tmp = defaultdict(list)
    for i, t in enumerate(decoded):
        tmp[t].append(i)
    for x in tmp:
        assert (
            len(tmp[x]) == 1
        ), f'surface form {x!r} maps to more than one token> {tmp[x]}'

    return decoded


###### The following code was taken from https://github.com/epfl-dlab/transformers-CFG/blob/main/transformers_cfg/tokenization/mapping.py


class Mapping:
    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.tokenizer = tokenizer
        self.special = tokenizer.all_special_ids
        self._length = len(self.tokenizer.get_vocab())

    def __len__(self):
        return self._length

    def _map(self, token_id: int) -> str:
        # if token_id is tensor, convert it to int
        if hasattr(token_id, 'item'):
            token_id = token_id.item()
        raw_token = self.tokenizer.convert_ids_to_tokens(token_id)
        return raw_token

    def __call__(self, token_id: int) -> bytes:
        token = self._map(token_id)
        return bytes(token, 'utf-8')


class GPT2Mapping(Mapping):
    #    def __init__(self, *args, **kwargs):
    #        super().__init__(*args, **kwargs)

    def __call__(self, token_id: int) -> str:
        raw_token = super()._map(token_id)
        if raw_token.startswith('Ġ'):
            raw_token = raw_token.replace('Ġ', ' ')
        if raw_token.startswith('Ċ'):
            raw_token = raw_token.replace('Ċ', '\n')
        if raw_token.startswith('ĉ'):
            raw_token = raw_token.replace('ĉ', '\t')
        return raw_token

    def post_process(self, tokens):
        return tokens


class LLaMaMapping(Mapping):
    def __call__(self, token_id: int) -> str:
        raw_token = super()._map(token_id)
        raw_token = raw_token.replace('Ġ', ' ')
        raw_token = raw_token.replace('Ċ', '\n')
        raw_token = raw_token.replace('ĉ', '\t')
        raw_token = raw_token.replace('č', '\r')
        return raw_token

    def post_process(self, tokens):
        return tokens


class CodeLLaMaMapping(Mapping):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.last_token_id = None

    def __call__(self, token_id: int) -> str:
        raw_token = super()._map(token_id)

        # we need to check if the token is at the beginning of the sentence to remove the space
        # specific to BPE
        at_bos = False
        if self.last_token_id is not None and self.last_token_id == self.bos_token_id:
            at_bos = True
        self.last_token_id = token_id
        if raw_token.startswith('▁'):
            raw_token = raw_token.replace('▁', ' ')
            if at_bos:
                # remove space at the beginning of the sentence
                raw_token = raw_token[1:]
        return raw_token

    def post_process(self, tokens):
        decoded = []
        for t in tokens:
            if t.startswith('<0x'):
                new_t = chr(int(t[3:-1], 16))
                if new_t not in tokens:
                    decoded.append(new_t)
                else:
                    decoded.append(t)
            else:
                decoded.append(t)
        return decoded
