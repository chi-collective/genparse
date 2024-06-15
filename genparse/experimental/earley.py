from collections import defaultdict
from functools import lru_cache

from arsenal.datastructures.pdict import pdict

from genparse.cfglm import EOS, add_EOS
from genparse.linear import WeightedGraph
from genparse.lm import LM
from genparse.semiring import Boolean

#from genparse.experimental.earley0 import EarleyLM, Earley, Column
from genparse.experimental.earley1 import EarleyLM, Earley, Column
