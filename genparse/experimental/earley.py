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
        self.q_incomplete = deque([])
        self.q_complete = defaultdict(BucketQueue)

        # priority queue over positions I in `Column` K.
        self.q_j = BucketQueue()

        # set of nonterminals that have already been predicted in this column
        self.predicted = set()


class PredictFilter:

    def __init__(self, cfg):
        self.cfg = cfg

        R = defaultdict(set)
        P = defaultdict(set)  # parent table
        for r in cfg:
            A = r.head
            if len(r.body) == 0: continue
            B = r.body[0]
            R[A,B].add(r)
            P[B].add(A)
        self.P = P
        self.R = R

        self.prediction = {}
        for token in cfg.V:
            ancestors = self._ancestry(token)
            for B in ancestors:
                tmp = set()
                for A in ancestors[B]:
                    tmp.update(self.R[B, A])
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
    Class to run our O(N^3|G|) faster Earley's algorithm
    Warning: Assumes that nullary rules and unary chain cycles have been removed
    """

    __slots__ = ('cfg', 'order', 'col', '_predict_filter')

    def __init__(self, cfg):
        assert not cfg.has_nullary() and not cfg.has_unary_cycle()
        self.cfg = cfg
        self.col = None
        self.order = cfg._unary_graph().buckets
        self._predict_filter = PredictFilter(cfg)

    def __call__(self, sentence):
        N = len(sentence)

        # return if empty string
        if N == 0:
            return self.cfg.null_weight_start()

        zero = self.cfg.R.zero
        rhs = self.cfg.rhs

        # initialize bookkeeping structures
        self.col = []

        col = Column(0, self.cfg.R.chart())
        self._predict(col, self.cfg.S, sentence[0])
        self.col.append(col)

        for k in range(N+1):

            col = self.col[k]

            # proceed to next item set only if there are items waiting on the queue
            if len(col.q_incomplete) == 0 and len(col.q_complete) == 0:
                return zero

            while col.q_j:
                j = col.q_j.pop()
                col_j = self.col[j]

                # Attach the newly completed item to available customers:
                # missing(I, X/Ys, K) += missing(I, X/[Y|Ys], J) * complete(J, Y, K)
                Q = col.q_complete[j]
                while Q:
                    Y = Q.pop()
                    y = col.chart[j,Y]
                    for (I, X, Ys) in col_j.waiting_for[Y]:
                        self._update(col, I, X, Ys[1:], col_j.chart[I,X,Ys] * y)

            if k < N:

                token = sentence[k]

                next_col = Column(k+1, self.cfg.R.chart())
                self.col.append(next_col)

                Q = col.q_incomplete
                while Q:
                    (I, X, Ys) = Q.popleft()
                    Y = Ys[0]
                    if self.cfg.is_terminal(Y):
                        # SCAN
                        # missing(I, X/Ys, K) += missing(I, X/[Y|Ys], J) * word(J, Y, K)
                        if Y == token:
                            self._update(next_col, I, X, Ys[1:], col.chart[I, X, Ys])
                    else:
                        # PREDICT
                        # missing(K, X/Ys, K) += rule(X -> Ys) needed only if needs(X, K), is_leftcorner(X, W), word(K, W, K)
                        self._predict(col, Y, token)

        return self.col[N].chart[0, self.cfg.S]

    def _predict(self, col, X, token):
        if X in col.predicted: return
        col.predicted.add(X)
        zero = self.cfg.R.zero
        if token not in self.cfg.V: return
        for r in self._predict_filter(token, X):
            self._update(col, col.k, X, r.body, r.w)

    def _update(self, col, I, X, Ys, value):
        k = col.k
        if Ys == ():
            # Items of the form missing(I, X/[], K)
            was = col.chart[I,X]
            if was == self.cfg.R.zero:
                col.q_j[I] = k if I == k else (k-I-1)
                col.q_complete[I][X] = self.order[X]
            col.chart[I,X] = was + value

        else:
            # Items of the form missing(I, X/[Y|Ys], K)
            item = (I, X, Ys)
            was = col.chart[item]
            if was == self.cfg.R.zero:
                col.q_incomplete.append(item)
                col.waiting_for[Ys[0]].append(item)
            col.chart[item] = was + value
