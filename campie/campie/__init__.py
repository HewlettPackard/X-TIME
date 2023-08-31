"""
Python APIs to simulate various CAMs (Content Addressable Memory) on GPUs at scale.
"""

import importlib.metadata

from .cam import (
    acam_count_mismatches,
    acam_match,
    acam_reduce_sum,
    tcam_hamming_distance,
    tcam_match,
    tcam_reduce_sum,
)
from .util import flip_indices

__all__ = [
    "acam_count_mismatches",
    "acam_match",
    "acam_reduce_sum",
    "flip_indices",
    "tcam_hamming_distance",
    "tcam_match",
    "tcam_reduce_sum",
]

# https://github.com/python-poetry/poetry/issues/144#issuecomment-1488038660
__version__ = importlib.metadata.version("campie")
