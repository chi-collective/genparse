def prefix_closure(C):
    """
    Compute prefix closure of `C`, including epsilon.

      >>> prefix_closure(['AA', 'ABB'])
      ['', 'A', 'AA', 'AB', 'ABB']

    If you pass in strings, you get strings back

      >>> prefix_closure(['B', 'A', 'BB'])
      ['', 'A', 'B', 'BB']

    If you pass in tuples, you get tuples back.

      >>> prefix_closure([('B',), ('A',), ('B','B')])
      [(), ('A',), ('B',), ('B', 'B')]

    """
    P = {p for z in C for p in prefixes(z)}
    return list(sorted(P))


def prefixes(z):
    """
    Return the prefixes of the sequence `z`

      >>> list(prefixes(''))
      ['']

      >>> list(prefixes('abc'))
      ['', 'a', 'ab', 'abc']

    """
    for p in range(len(z)+1):
        yield z[:p]


def suffixes(z):
    """
    Return the prefixes of the sequence `z`

      >>> list(suffixes(''))
      ['']

      >>> list(suffixes('abc'))
      ['', 'c', 'bc', 'abc']

    """
    for p in reversed(range(len(z)+1)):
        yield z[p:]


def last_char_sub_closure(sigma, C):
    """Take the closure of `C` under last character substitution from the alphabet
    `sigma`.

    >>> last_char_sub_closure('ABC', [()])
    [('A',), ('B',), ('C',)]

    >>> last_char_sub_closure('ABC', [('A',)])
    [('A',), ('B',), ('C',)]

    >>> last_char_sub_closure('ABC', [('A','B')])
    [('A', 'A'), ('A', 'B'), ('A', 'C')]

    """
    return list(sorted({c[:-1] + (a,) for c in C for a in sigma}))


def longest_suffix_in(s, S):
    """
    The longest suffix of `s` that is in `S`.

    >>> longest_suffix_in('abcde', ['e', 'de'])
    'de'

    >>> longest_suffix_in('abcde', [''])
    ''

    """

    if s in S:
        return s
    elif not s:
        raise KeyError('no suffixes found')
    else:
        return longest_suffix_in(s[1:], S)


def max_munch(tokens):
    """
    Maximum-munch tokenizer for the finite set `tokens`.

      >>> tokens = ['aaa', 'aa', 'a']
      >>> t = max_munch(tokens)
      >>> assert t('aaaaaa') == ('aaa','aaa')
      >>> assert t('aaaaa') == ('aaa','aa')
      >>> assert t('aaaa') == ('aaa','a')
      >>> assert t('aaa' 'aaa' 'aa') == ('aaa','aaa', 'aa')

    """
    def t(a):
        if len(a) == 0: return ()
        for b in sorted(tokens, key=len, reverse=True):
            if a[:len(b)] == b:
                return (b,) + t(a[len(b):])
        raise ValueError('bad token set')
    return t
