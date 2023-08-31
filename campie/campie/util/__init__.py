"""
This module includes additional utility functionality related to CAMs,
implemented as CUDA kernels.
"""

from .flip_indices import flip_indices

__all__ = [
    "flip_indices",
]
