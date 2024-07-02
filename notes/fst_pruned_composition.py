"""
Coarse-grained pruning to improve the size of intermediate FST-FST
composition.
"""

import numpy as np
from time import time
from arsenal import colors, timers

from genparse import Float
from genparse.fst import EPSILON, FST
from genparse.segmentation import bpe_wfst

bpe_medium = [
    (26361, 'zon'),
    (14032, ' purple'),
    (33439, ' MIDI'),
    (43499, 'announced'),
    (33034, ' hardened'),
    (18411, ' thermal'),
    (42034, 'devices'),
    (42415, ' ankles'),
    (44356, ' Turtles'),
    (14886, ' Tyler'),
    (42111, ' infiltrate'),
    (46641, ' vividly'),
    (22999, ' burnt'),
    (5973, 'icken'),
    (817, 'Th'),
    (32498, ' clones'),
    (42535, '}}}'),
    (20933, ' warehouse'),
    (31641, ' Rw'),
    (3200, ' secret'),
    (15114, ' clouds'),
    (24081, ' Pav'),
    (40220, '1973'),
    (45577, ' Lauderdale'),
    (22846, 'ANN'),
    (9409, 'isd'),
    (12374, 'ettings'),
    (18180, ' rivers'),
    (25353, ' Background'),
    (14977, ' arbitrary'),
    (15545, ' Install'),
    (48465, ' microbiome'),
    (35094, ' Gau'),
    (40935, ' Daesh'),
    (17410, 'Method'),
    (41589, 'Harris'),
    (8975, ' Sometimes'),
    (29173, '644'),
    (12187, ' cake'),
    (34909, ' disabling'),
    (4849, ' Sal'),
    (22507, ' Allow'),
    (34480, ' Colony'),
    (2439, ' Sm'),
    (709, 'ov'),
    (47261, 'wat'),
    (16069, ' ranges'),
    (13431, ' frames'),
    (39911, ' Lov'),
    (35905, ' infuri'),
    (32630, ' ivory'),
    (24709, ' Speech'),
    (11258, 'arer'),
    (33994, '>('),
    (2135, 'ift'),
    (30433, 'utsche'),
    (22318, '375'),
    (24536, ' metro'),
    (45772, ' Byzantine'),
    (15688, 'sky'),
    (37748, ' layered'),
    (20047, ' emphasized'),
    (33738, ' righteousness'),
    (5459, 'ocket'),
    (23699, ' Carbon'),
    (43382, ' Vapor'),
    (1360, 'gy'),
    (21732, ' trophies'),
    (47915, '561'),
    (29532, 'facing'),
    (50004, ' Collections'),
    (4568, ' activities'),
    (7943, ' Arizona'),
    (6749, 'ician'),
    (13003, 'Added'),
    (23898, ' recons'),
    (33516, ' Tek'),
    (25367, 'brow'),
    (33725, ' Saiyan'),
    (6828, 'ounter'),
    (28496, 'aughty'),
    (4818, ' dat'),
    (35954, ' Deadline'),
    (7171, ' Series'),
    (3529, 'ero'),
    (35051, 'reflect'),
    (49377, ' topple'),
    (13834, ' proc'),
    (19526, '�'),
    (5369, ' identity'),
    (10936, ' Gary'),
    (38232, ' onward'),
    (20231, 'Address'),
    (14594, ' Working'),
    (14063, ' Yam'),
    (38632, ' ejected'),
    (13941, ' avoided'),
    (15929, ' Neil'),
    (21981, ' angels'),
    (5055, ' standing'),
    (45439, ' 406'),
    (46811, ' Townsend'),
    (29225, ']."'),
    (16635, 'Bo'),
    (4826, 'Ab'),
    (15518, ' Cand'),
    (67, 'd'),
    (23236, ' purported'),
    (37238, ' skirm'),
    (31175, '>:'),
    (46280, 'swe'),
    (27180, 'lake'),
    (27464, ' scaled'),
    (19831, 'itsu'),
    (34917, ' Freeze'),
    (42131, 'edited'),
    (35047, ' Cooperation'),
    (16813, 'iberal'),
    (30996, ' peacefully'),
    (16288, 'heimer'),
    (15598, ' sentiment'),
    (26151, ' ingest'),
    (13927, 'wear'),
    (28729, ' OVER'),
    (30256, ' loser'),
    (29836, ' asshole'),
    (15496, 'Hello'),
    (3667, ' claims'),
    (42303, ',...'),
    (43455, ' stressing'),
    (14610, ' -->'),
    (13504, ' exploring'),
    (43144, ' dipping'),
    (37156, ' Chero'),
    (9974, ' Drive'),
    (4109, ' Rober'),
    (35234, 'wives'),
    (1084, 'min'),
    (16537, ' lately'),
    (41843, ' valves'),
    (6967, ' collabor'),
    (46556, '088'),
    (35702, 'ック'),
    (15613, ' constitute'),
    (9154, ' tape'),
    (21994, ' astonishing'),
    (4070, 'azing'),
    (24269, ' Trent'),
    (42375, ' Angle'),
    (21546, ' therapeutic'),
    (8822, 'eland'),
    (23833, 'alks'),
    (10857, ' Temple'),
    (37689, 'holiday'),
    (39763, ' ~/.'),
    (20129, ' java'),
    (11440, ' Ident'),
    (4392, ' Syria'),
    (47598, 'Crash'),
    (47223, 'owsky'),
    (22681, ' ballistic'),
    (22025, ' Sto'),
    (16386, ' Experience'),
    (30898, 'embedreportprint'),
    (30699, ' hardship'),
    (41632, ' believable'),
    (22201, ' incarn'),
    (17208, 'arse'),
    (9272, ' gathered'),
    (46618, ' 413'),
    (34324, ' indifference'),
    (20821, ' snapped'),
    (22226, ' paradox'),
    (19303, ' interim'),
    (10287, ' les'),
    (17468, ' Bennett'),
    (3264, ' directly'),
    (4369, ' disease'),
    (4885, ' Western'),
    (14804, '=>'),
    (40323, ' Kut'),
    (39081, 'administ'),
    (33749, ' derail'),
    (15797, ' trim'),
    (10889, ' Sure'),
    (18330, ' inspire'),
    (45928, 'flying'),
    (3134, '67'),
    (41512, ' Err'),
    (7967, ' Snow'),
    (30543, '"\''),
    (33244, 'Success'),
    (40145, ' JPM'),
    (11185, ' meets'),
    (32571, ' Rohing'),
    (2132, 'ully'),
    (6462, ' Full'),
    (29519, 'ه'),
    (31975, 'warm'),
    (28740, ' calorie'),
]


bpe_small = [
    (26361, 'zon'),
    (14032, ' purple'),
    (33439, ' MIDI'),
    (43499, 'announced'),
    (33034, ' hardened'),
    (18411, ' thermal'),
    (42034, 'devices'),
    (42415, ' ankles'),
    (44356, ' Turtles'),
    (14886, ' Tyler'),
    (42111, ' infiltrate'),
    (46641, ' vividly'),
    (22999, ' burnt'),
    (5973, 'icken'),
    (817, 'Th'),
    (32498, ' clones'),
    (42535, '}}}'),
    (20933, ' warehouse'),
    (31641, ' Rw'),
    (3200, ' secret'),
]


TIMER = timers()


class CoarseCompositionFilter:
    def __init__(
        self,
        a2b,
        b2c,
        N1=lambda x: x,
        A=lambda x: x,
        B=lambda x: x,
        C=lambda x: x,
        N2=lambda x: x,
    ):
        # WARNING: the transducers need to be epsilon-free in the correct ways to use _compose!

        self.A = A
        self.B = B
        self.C = C
        self.N1 = N1
        self.N2 = N2

        self.a2b = a2b
        self.b2c = b2c

        before = time()
        with TIMER['coarse-build']:
            self.coarse_a2b = a2b.coarsen(self.N1, self.A, self.B)
            self.coarse_b2c = b2c.coarsen(self.N2, self.B, self.C)

        with TIMER['coarse-compose']:
            self.tmp = self.coarse_a2b._compose(
                self.coarse_b2c,
                # coarsen=False,
                coarsen=True,  # coarse to fine!
            )

        with TIMER['coarse-trim']:
            self.coarse_a2c = self.tmp.trim  # Trimmed!!!

        self.took = time() - before

        self.coarse_arcs = {(i, (a, c), j) for i, (a, c), j, _ in self.coarse_a2c.arcs()}

    def coarse_arc(self, i, label, j):
        a, c = label
        return (self.coarse_state(i), (self.A(a), self.C(c)), self.coarse_state(j))

    def keep_arc(self, i, label, j):
        return self.coarse_arc(i, label, j) in self.coarse_arcs

    def coarse_state(self, x):
        p, q = x
        return (self.N1(p), self.N2(q))

    @staticmethod
    def fidelity(a2b, b2c, keep, filter_time):
        "This method is for introspection.  It measures the quality of the filter."

        before = time()
        fine = a2b._pruned_compose(
            b2c, keep=lambda x: True, keep_arc=lambda i, label, j: True
        )
        fine.trim
        took = time() - before

        # print(fine.states)
        # print(fine.trim.states)
        # print(self.coarse_a2c.states)

        # approximately filtered states
        F = {state for state in fine.states if keep(state)}

        # perfectly trimmed states
        T = fine.trim.states

        # precision = len(F & T) / len(F) if len(F) > 0 else 1
        recall = len(F & T) / len(T) if len(T) > 0 else 1

        print()
        # print(colors.yellow % 'Statistics')
        # print(colors.yellow % '==========')
        # print(colors.yellow % 'precision:', precision)
        # print(colors.yellow % 'recall:   ', recall)
        # print(colors.yellow % 'overlap: ', len(F & T))
        # print(colors.yellow % 'trimmed: ', len(T))
        # print(colors.yellow % 'filtered:', len(F))

        print(colors.yellow % 'states:', len(T), '≤', len(F), '≤', len(fine.states))
        print(
            colors.yellow % 'filter overhead:',
            (
                colors.green % (filter_time / took)
                if filter_time < took
                else colors.light.red % (filter_time / took)
            ),
            filter_time,
            took,
        )

        print(
            1 - (len(F) - len(T)) / (len(fine.states) - len(T))
            if (len(fine.states) - len(T)) > 0
            else 1
        )

        # correctness test: make sure that we keep all of the trimmed states
        for state in fine.trim.states:
            assert keep(state), state
        assert recall == 1

        # print(colors.yellow % 'arcs....')

    #        for (i1,i2), (a,c), (j1,j2), _ in fine.trim.arcs():
    #            arc = (i1,i2), (a,c), (j1,j2)
    #            coarse_arc = self.coarse_arc((i1,i2), (a, c), (j1,j2))
    #            #print(colors.mark(coarse_arc in self.coarse_arcs))
    #            #print(' ', arc)
    #            #print(' ', coarse_arc)
    #            assert coarse_arc in self.coarse_arcs

    # TIMER.compare()

    @staticmethod
    def show_timer():
        Z = sum(T.mean for k, T in TIMER.items())
        for k, T in TIMER.items():
            print(f'{T.mean / Z * 100:.2f}%: {k}, {T.mean}')

    def __call__(self, state):
        "Should we keep this fine-grained state?"
        return self.coarse_state(state) in self.coarse_a2c.states


# XXX: Be careful - epsilon cannot be merged like the other labels (that should
# probably be enforced in the coarsen method).


def random_hash(domain):
    if domain >= 2:
        memo = {}

        def f(x):
            y = memo.get(x)
            if y is None:
                memo[x] = y = int(np.random.randint(0, domain)) if x != EPSILON else x
            return y

        return f

    else:
        return lambda x: x


# XXX: the construction does not yet support epsilons!
def test_bpe():
    if 1:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        print(f'token vocabulary size: {tokenizer.vocab_size}')
        T = [
            (token_id, tokenizer.decode([token_id]))
            for token_id in range(tokenizer.vocab_size)
        ]

        import random

        random.seed(8675309)
        # S = random.sample(T, 200)
        S = T

    else:
        S = bpe_medium
        # S = bpe_small

    benchmark = timers()

    b2c = bpe_wfst(S).renumber
    c2c = FST.from_string('zon purple secret infiltrate', Float).renumber

    print(colors.cyan % colors.line(80))
    print(colors.cyan % 'unpruned')
    FST.PRUNING = None
    with benchmark['unpruned']:
        full = b2c @ c2c
    print(
        colors.yellow % 'states:',
        len(full.states),
        len(full.trim.states),
        colors.yellow % 'arcs:',
        len(list(full.arcs())),
        len(list(full.trim.arcs())),
    )

    filters = []

    def pruning(self, other):
        sizes = (
            len(self.states),
            len(other.states),
            # len(self.A), len(self.B), len(other.A), len(other.B)
        )
        size = min(sizes)

        buckets = 20

        if size > buckets:
            print(colors.red % '>>>', size, ':::', sizes)

            cond = len(self.states) <= len(other.states)

            keep = CoarseCompositionFilter(
                self,
                other,
                #                N1 = random_hash(len(self.states) // 2) if len(self.states) <= len(other.states) else lambda x: x,
                N1=random_hash(buckets) if cond else lambda x: x,
                # A = random_hash(len(self.A) // 2),
                # B = random_hash(len(self.B) // 2),
                # B = random_hash(10),
                # C = random_hash(len(other.B) // 2),
                #                N2 = random_hash(len(other.states) // 2) if len(self.states) > len(other.states) else lambda x: x,
                N2=random_hash(buckets) if not cond else lambda x: x,
                #                N2 = random_hash(buckets),
            )

            filters.append(keep)

            return keep

        else:
            keep = lambda x: True
            keep.keep_arc = lambda i, label, j: True
            return keep

    print(colors.cyan % colors.line(80))
    print(colors.cyan % 'pruned')
    FST.PRUNING = pruning
    with benchmark['pruned']:
        have = b2c @ c2c

    print(
        colors.yellow % 'states',
        len(have.trim.states),
        '<=',
        len(have.states),
        colors.yellow % 'arcs',
        len(list(have.trim.arcs())),
        '<=',
        len(list(have.arcs())),
    )

    for keep in filters:
        CoarseCompositionFilter.fidelity(keep.a2b, keep.b2c, keep, keep.took)

    print()
    CoarseCompositionFilter.show_timer()

    print(colors.cyan % colors.line(80))
    benchmark.compare()


def test_basic1():
    a2b = FST(Float)
    a2b.add_I('#', 1.0)
    a2b.add_arc('#', ('a', 'b'), '#', 0.5)
    a2b.add_F('#', 1.0)

    assert a2b('aaa', 'bbb') == 0.5**3

    b2c = FST(Float)
    b2c.add_I('s0', 1.0)
    b2c.add_arc('s0', ('b', 'c'), 's0', 0.5)
    b2c.add_F('s0', 1.0)

    print(a2b.states, b2c.states)

    print(colors.magenta % 'fine-grained')
    before = time()
    keep = CoarseCompositionFilter(a2b, b2c)
    filter_time = time() - before
    CoarseCompositionFilter.fidelity(a2b, b2c, keep, filter_time)

    print(colors.magenta % 'super-coarse-grained')
    before = time()
    keep = CoarseCompositionFilter(
        a2b,
        b2c,
        N1=lambda x: '#',
        A=lambda x: '$',
        B=lambda x: '$',
        C=lambda x: '$',
        N2=lambda x: '#',
    )
    filter_time = time() - before
    CoarseCompositionFilter.fidelity(a2b, b2c, keep, filter_time)


if __name__ == '__main__':
    from arsenal import testing_framework

    testing_framework(globals())
