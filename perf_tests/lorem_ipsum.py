# From http://www.catipsum.com/index.php
LOREM_IPSUM_GRAMMAR = r"""
start: "Howl on top of tall thing hiding behind the couch until lured out by a feathery toy purr" \
    "when owners are asleep, cry for no apparent reason and meow meow you are my owner so here is" \
    "dead rat. Cough hairball, eat toilet paper stick butt in face, but don't nosh on the birds" \
    "slap the dog because cats rule for catasstrophe yet kitty scratches couch bad kitty pet my" \
    "you know you want to; seize the hand and shred it!"
"""


_LINE_LENGTH_CAP = 100
_INDENT_SIZE = 4


def ipsum_to_grammar(text: str) -> str:
    lines = []
    line_words = []
    for word in text.split():
        prefix = 'start: ' if not lines else ' ' * _INDENT_SIZE
        line_length = (
            # there are `len(line_words) - 1` spaces, plus 4 characters for the
            # leading quote and the trailing `" \`.
            sum(len(word) for word in line_words) + len(line_words) + 3 + len(prefix)
        )
        if line_words and line_length + len(word) + 1 > _LINE_LENGTH_CAP:
            line = prefix + '"' + ' '.join(line_words) + '" \\'
            lines.append(line)
            line_words = []
        else:
            line_words.append(word)
    if line_words:
        prefix = 'start: ' if not lines else ' ' * _INDENT_SIZE
        line = prefix + '"' + ' '.join(line_words) + '" \\'
        lines.append(line)
    result: str = '\n'.join(lines).rstrip('\\')
    return result
