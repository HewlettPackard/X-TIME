"""Helper functionality."""

from math import ceil

from ..types import LaunchConfiguration


def simple_kernel_dimensions(
    threads_needed: int, max_threads_per_block: int
) -> LaunchConfiguration:
    """
    Determines a simple one-dimensional set of CUDA kernel dimensions,
    i.e. `(dim_grid, dim_block)` based on a wanted amount of threads
    """

    if threads_needed <= 0:
        return ((0, 0, 0), (0, 0, 0))

    threads_per_block = min(threads_needed, max_threads_per_block)
    blocks_per_grid = ceil(threads_needed / threads_per_block)

    dim_grid = (blocks_per_grid, 1, 1)
    dim_block = (threads_per_block, 1, 1)

    return dim_grid, dim_block
