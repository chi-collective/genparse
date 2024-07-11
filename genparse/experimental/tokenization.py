"""
Tools for tokenization.
"""

from arsenal import defaultdict
from transformers import AutoTokenizer
import re
import warnings


def decode_tokenizer_vocab(tokenizer):
    name = tokenizer.name_or_path.lower()
    if 'gpt2' in name:
        decoder = GPT2Decoder(tokenizer)
    elif 'codellama' in name:
        decoder = CodeLLaMaDecoder(tokenizer)
    elif 'llama-3' in name:
        decoder = LLaMa3Decoder(tokenizer)
    else:
        warnings.warn('baaaadd')
        decoder = VocabularyDecoder(tokenizer)

    decoded = decoder.decode_vocab()

    tmp = defaultdict(list)
    for i, t in enumerate(decoded):
        tmp[t].append(i)
    for x in tmp:
        # assert (
        #    len(tmp[x]) == 1
        # ), f'surface form {x!r} maps to more than one token> {tmp[x]}'
        if not len(tmp[x]) == 1:
            warnings.warn(f'surface form {x!r} maps to more than one token> {tmp[x]}')

    return decoded


class VocabularyDecoder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.byte_tokens = [b''] * len(self.tokenizer)

        if hasattr(tokenizer, 'byte_decoder'):
            self.byte_decoder = self.tokenizer.byte_decoder
            self._decode_with_byte_decoder()
        elif hasattr(tokenizer, 'sp_model'):
            self._decode_with_sp_model()
        else:  # fallback to gpt2
            self.byte_decoder = AutoTokenizer.from_pretrained(
                'gpt2', use_fast=False
            ).byte_decoder

            self.byte_decoder[' '] = 32
            self.byte_decoder['\n'] = 10
            self.byte_decoder['\r'] = 13
            self.byte_decoder['\t'] = 9
            self.byte_decoder['▁'] = 32

            self._decode_with_byte_decoder()

    def _decode_with_byte_decoder(self):
        for i in range(len(self.tokenizer)):
            byte_coded = bytes(
                [self.byte_decoder[c] for c in self.tokenizer.convert_ids_to_tokens(i)]
            )
            self.byte_tokens[i] = byte_coded

    def _decode_with_sp_model(self):
        space_prefix = '▁'.encode()
        special_tokens_map = {
            id: token for token, id in self.tokenizer.get_added_vocab().items()
        }
        for i in range(len(self.tokenizer)):
            if i in special_tokens_map:
                byte_coded = special_tokens_map[i].encode()
            else:
                byte_coded = re.sub(
                    rb'<0x(..)>',
                    lambda x: bytes.fromhex(x[1].decode()),
                    self.tokenizer.sp_model.id_to_piece(i).encode(),
                )
            self.byte_tokens[i] = byte_coded.replace(space_prefix, b' ')

    def _map(self, token_id):
        if hasattr(token_id, 'item'):
            token_id = token_id.item()
        raw_token = self.byte_tokens[token_id]
        return raw_token

    def map(self, token_id):
        raw_token = self._map(token_id)
        try:
            return raw_token.decode('utf-8')
        except UnicodeDecodeError:
            fallback = self.tokenizer.convert_ids_to_tokens(token_id)
            warnings.warn(
                f'Cannot decode {repr(raw_token)} (UnicodeDecodeError).'
                f'Falling back to {fallback} with tokenizer.convert_ids_to_tokens.'
            )
            return fallback

    def _post_process_vocab(self, vocab):
        return vocab

    def decode_vocab(self):
        return self._post_process_vocab([self.map(i) for i in range(len(self.tokenizer))])


class GPT2Decoder(VocabularyDecoder):
    def __init__(self, tokenizer):
        if tokenizer.is_fast:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer.name_or_path, use_fast=False
            )
        super().__init__(tokenizer=tokenizer)


class LLaMa3Decoder(VocabularyDecoder):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)


class CodeLLaMaDecoder(VocabularyDecoder):
    def __init__(self, tokenizer):
        if tokenizer.is_fast:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer.name_or_path, use_fast=False
            )
        super().__init__(tokenizer=tokenizer)
