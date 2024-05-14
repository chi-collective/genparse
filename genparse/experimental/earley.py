from collections import defaultdict, deque, namedtuple
from queue import PriorityQueue
from arsenal.datastructures.bucketqueue import BucketQueue


class Column:
    def __init__(self, k, chart):

        self.k = k

        # item values
        self.chart = chart

        # track the left-most missing item
        self.waiting_for = defaultdict(list)

        # separate queues for complete and incomplete items
        self.q_complete = defaultdict(BucketQueue)

        # priority queue over positions I in `Column` K.
        self.q_j = BucketQueue()

        # set of nonterminals that have already been predicted in this column
        self.predicted = set()


class PredictFilter:

    def __init__(self, cfg):
        self.cfg = cfg

        R = defaultdict(list)
        P = defaultdict(set)  # parent table
        for r in cfg:
            A = r.head
            if len(r.body) == 0: continue
            B = r.body[0]
            R[A,B].append(r)
            P[B].add(A)
        self.P = P
        self.R = R

        self.prediction = {}
        for token in cfg.V:
            ancestors = self._ancestry(token)
            for B in ancestors:
                tmp = []
                for A in ancestors[B]:
                    tmp.extend(self.R[B, A])
                self.prediction[token, B] = tmp

    def _ancestry(self, w):
        S = defaultdict(set)
        stack = [w]
        while stack:
            Y = stack.pop(0)
            for X in self.P[Y]:
                if Y not in S[X]:
                    stack.append(X)
                    S[X].add(Y)
        return S

    def __call__(self, token, B):
        return self.prediction.get((token, B), [])


class Earley:
    """
    Implements a semiring-weighted version Earley's algorithm that runs in O(N^3|G|) time.
    Warning: Assumes that nullary rules and unary chain cycles have been removed
    """

    __slots__ = ('cfg', 'order', 'col', '_predict_filter')

    def __init__(self, cfg):
        assert not cfg.has_nullary() and not cfg.has_unary_cycle()
        self.cfg = cfg
        self.col = None
        self.order = cfg._unary_graph_transpose().buckets
        self._predict_filter = PredictFilter(cfg)

    def __call__(self, sentence):
        N = len(sentence)

        # return if empty string
        if N == 0:
            return self.cfg.null_weight_start()

        # initialize bookkeeping structures
        self.col = [Column(0, self.cfg.R.chart())]
        self.col[0].predicted.add(self.cfg.S)
        for r in self._predict_filter(sentence[0], self.cfg.S):
            self._update(self.col[0], 0, r.head, r.body, r.w)

        self.PREDICT(self.col[0], sentence[0])

        for k in range(N):
            self.col.append(self.next_column(self.col[k], sentence[k], sentence[k+1] if k+1 < len(sentence) else None))

        return self.col[N].chart[0, self.cfg.S]

    def next_column(self, prev_col, token, next_token):

        next_col = Column(prev_col.k + 1, self.cfg.R.chart())

        # SCAN: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y, K)
        for I, X, Ys in prev_col.waiting_for[token]:
            self._update(next_col, I, X, Ys[1:], prev_col.chart[I, X, Ys])

        # ATTACH: phrase(I, X/Ys, K) += phrase(I, X/[Y|Ys], J) * phrase(J, Y/[], K)
        while next_col.q_j:
            j = next_col.q_j.pop()
            col_j = self.col[j]
            Q = next_col.q_complete[j]
            while Q:
                Y = Q.pop()
                y = next_col.chart[j,Y]
                for (I, X, Ys) in col_j.waiting_for[Y]:
                    self._update(next_col, I, X, Ys[1:], col_j.chart[I,X,Ys] * y)

        # PREDICT (based on one step of lookahead)
        if next_token is not None:
            self.PREDICT(next_col, next_token)

        return next_col

    def PREDICT(self, prev_col, token):
        # PREDICT: phrase(K, X/Ys, K) += rule(X -> Ys) with lookahead to prune
        Q = deque(list(prev_col.waiting_for))
        while Q:
            X = Q.popleft()
            if self.cfg.is_terminal(X) or X in prev_col.predicted: continue
            prev_col.predicted.add(X)
            for r in self._predict_filter(token, X):
                #self._update(prev_col, prev_col.k, Y, r.body, r.w)

                Y = r.body[0]
                item = (prev_col.k, X, r.body)
                was = prev_col.chart[item]
                if was == self.cfg.R.zero:
                    prev_col.waiting_for[Y].append(item)
                    Q.append(Y)
                prev_col.chart[item] = was + r.w

    def _update(self, col, I, X, Ys, value):
        k = col.k
        if Ys == ():
            # Items of the form phrase(I, X/[], K)
            was = col.chart[I,X]
            if was == self.cfg.R.zero:
                col.q_j[I] = k if I == k else (k-I-1)
                col.q_complete[I][X] = self.order[X]
            col.chart[I,X] = was + value

        else:
            # Items of the form phrase(I, X/[Y|Ys], K)
            item = (I, X, Ys)
            was = col.chart[item]
            if was == self.cfg.R.zero:
                col.waiting_for[Ys[0]].append(item)
            col.chart[item] = was + value
