import dataclasses
import json
import math
import pathlib

import torch

from .canonicalizer import (
    NonCanonicalTokenizationError,
    CanonicalizerIterator,
    Canonicalizer,
)


@dataclasses.dataclass
class BPECanonicalizerIterator(CanonicalizerIterator):
    parent: 'BPECanonicalizer'
    state: int

    def next(self, token):
        new_state = self.parent.transitions.get((self.state, token))
        if new_state is None:
            raise NonCanonicalTokenizationError
        self.state = new_state

    def mask(self):
        return self.parent.mask_tensor[self.state]


@dataclasses.dataclass
class BPECanonicalizer(Canonicalizer):
    transitions: dict[tuple[int, int], int]
    mask_tensor: torch.Tensor
    _eos_token_id: int

    def iterator(self):
        return BPECanonicalizerIterator(self, 0)

    def eos_token_id(self):
        return self._eos_token_id

    @staticmethod
    def from_file(
        path: pathlib.Path, dtype: torch.dtype, device: torch.device
    ) -> 'BPECanonicalizer':
        data = torch.load(path)
        eos_token_id = data['eos_token_id']
        transition_list = data['transitions'].tolist()
        # Index the transitions by source state and symbol.
        transitions = {
            (state_from, symbol): state_to
            for state_from, symbol, state_to in transition_list
        }
        # Precompute the mask tensors.
        mask_tensor = torch.full(
            (data['num_states'], data['vocabulary_size']),
            -math.inf,
            dtype=dtype,
            device=device,
        )
        # Conveniently, the keys of the transitions dict are the coordinates
        # of the entries that should be 0.
        index_tensor = torch.tensor(list(transitions.keys()), device=device)
        mask_tensor[torch.unbind(index_tensor, dim=1)] = 0
        # All states are accept states, so EOS is allowed at every state.
        mask_tensor[:, eos_token_id] = 0
        return BPECanonicalizer(transitions, mask_tensor, eos_token_id)
