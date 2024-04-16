from genparse.fst import FST, EPSILON
from genparse import Float
from arsenal import iterview
from collections import Counter


def bpe_wfst(S):
    m = FST(Float)
    START = 0
    STOP = 1
    m.add_I(0, 1)
    for i, x in S:
        m.add_arc(START, (i, EPSILON), (i, 0), 1)
        for j in range(len(x)):
            m.add_arc((i,j), (EPSILON, x[j]), (i,j+1), 1)
        m.add_arc((i,len(x)), (EPSILON, EPSILON), STOP, 1)
    m.add_F(STOP, 1)
    m.add_arc(STOP, (EPSILON, EPSILON), START, .1)   # decay
    return m




from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f'token vocabulary size: {tokenizer.vocab_size}')
T = [(token_id, tokenizer.decode([token_id])) for token_id in range(tokenizer.vocab_size)]

import random
random.seed(8675309)
#S = random.sample(T, 10_000)
S = T

T_b_c = bpe_wfst(S)



tmp = T_b_c.renumber.project(1)

# epsilon elimination step
E = tmp.E


print('block sizes:', Counter(len(B) for B in list(E.blocks())))

W = E.E
star = E.WeightType.star

for B in iterview(list(E.blocks())):

    if len(B) > 1:
        # this is pretty slow...
        b = E._closure(E.E, B)
    else:
        [i] = B
        tmp = {(i,i): star(W[i,i])}

#    print(B)
#    print(b)

#    break





# from arsenal import Integerizer
# for B in iterview(list(E.blocks())):
#     f = Integerizer()
#     tmp = np.zeros((len(B), len(B)))
#     for i in B:
#         for j in B:
#             tmp[f(i), f(j)] += E.E[i,j]
#     foo = np.linalg.inv(np.eye(len(B)) - tmp)
#     sol = Counter()
#     for i in range(len(B)):
#         for j in range(len(B)):
#             sol[f[i], f[j]] = sol[i,j]
