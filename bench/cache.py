from collections import deque
import logging
import pickle
import os
from genparse.util import lark_guide
from genparse.experimental.batch_inference import (
    ParallelCharacterProposal,
    ParallelTokenProposal,
)
import psutil
import warnings


class ProposalCache:
    def __init__(self, guide_cache_path, maxsize=10, memory_thresh=80):
        self.maxsize = maxsize
        self.cache = {}
        self.guide_cache = PersistentGuideCache(guide_cache_path)
        self.recent_keys = deque(maxlen=10)
        self.base_usage = psutil.virtual_memory().percent
        self.memory_thresh = memory_thresh

    def make_cache_key(self, grammar, proposal_name, proposal_args):
        key = [grammar, proposal_name]

        for arg in sorted(proposal_args):
            key.append(proposal_args[arg])

        return tuple(key)

    def fetch_or_create_proposal(
        self,
        llm,
        grammar,
        proposal_name,
        n_processes,
        proposal_args={},
        max_n_particles=250,
    ):
        # current_mem_usage = psutil.virtual_memory().percent
        # if current_mem_usage > self.memory_thresh:
        #    mem_usage = current_mem_usage
        #    while mem_usage > self.memory_thresh and len(self.cache) > 0:
        #        self.evict_object(self.recent_keys[-1])
        #        mem_usage = psutil.virtual_memory().percent

        key = self.make_cache_key(grammar, proposal_name, proposal_args)
        if key in self.cache:
            self.recent_keys.append(key)
            return self.cache[key]
        else:
            guide = self.guide_cache.get(grammar)

            if proposal_name == 'character':
                parallel_proposal = ParallelCharacterProposal(
                    llm=llm,
                    guide=guide,
                    num_processes=n_processes,
                    max_n_particles=max_n_particles,
                    seed=0,
                )
            elif proposal_name == 'token':
                parallel_proposal = ParallelTokenProposal(
                    llm=llm,
                    guide=guide,
                    K=proposal_args['K'],
                    num_processes=n_processes,
                    max_n_particles=max_n_particles,
                    seed=0,
                )
            else:
                raise ValueError(f'{proposal_name} is an invalid proposal name')

            self.cache[key] = parallel_proposal
            self.recent_keys.append(key)

            if len(self.cache) > self.maxsize:
                self.evict_objects()

            return parallel_proposal

    def evict_objects(self):
        keys_to_remove = set(self.cache.keys()) - set(self.recent_keys)
        for key in keys_to_remove:
            self.evict_object(key)

    def evict_object(self, key):
        self.cache[key].__del__()

        # XXX test this
        # used in case we need to evict objects to avoid exceed the memory threshold
        self.recent_keys = [k for k in self.recent_keys if k != key]

    def clear_cache(self):
        for key in self.cache.keys():
            self.evict_object(self, key)

        self.recent_keys.clear()
        self.cache = {}

    def __repr__(self):
        return f'ProposalCache(maxsize={self.maxsize}, current_size={len(self.cache)})'


class PersistentGuideCache:
    def __init__(self, filename):
        self.filename = filename
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def save_cache(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.cache, f)

    def get(self, grammar):
        if grammar in self.cache:
            return self.cache.get(grammar)
        else:
            guide = lark_guide(grammar)
            self.set(grammar, guide)
            return guide

    def set(self, key, value):
        self.cache[key] = value
        self.save_cache()

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]
            self.save_cache()
