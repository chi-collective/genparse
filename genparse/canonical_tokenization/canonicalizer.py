import math
from collections.abc import Iterable

import torch


class NonCanonicalTokenizationError(ValueError):
    pass


class CanonicalizerIterator:
    def next(self, token: int) -> None:
        """Read a token and update the state of this iterator accordingly.

        Raise NonCanonicalTokenizationError if this token would result in a
        non-canonical token sequence."""
        # Raises if the token sequence is invalid
        raise NotImplementedError

    def mask(self) -> torch.Tensor:
        """Return a mask representing which tokens may validly be generated
        next given the current state of this iterator.

        The result is a tensor with 0 at indexes for valid tokens, and -inf at
        indexes for invalid tokens.
        """
        raise NotImplementedError


class Canonicalizer:
    def masks(self, tokens: Iterable[int]) -> Iterable[torch.Tensor]:
        """Read n tokens and generate a sequence of n+1 masks.

        The eos_token index is needed to check that the token sequence is
        allowed to end when it does.
        """
        it = self.iterator()
        mask = it.mask()
        yield mask
        for token in tokens:
            it.next(token)
            mask = it.mask()
            yield mask
        if mask[self.eos_token_id()].item() == -math.inf:
            raise NonCanonicalTokenizationError('this token sequence cannot end with EOS')

    def iterator(self) -> CanonicalizerIterator:
        """Return a CanonicalizerIterator that can be used to compute masks
        iteratively, e.g., during decoding."""
        raise NotImplementedError

    def eos_token_id(self) -> int:
        """Get the id of the EOS token."""
        raise NotImplementedError
