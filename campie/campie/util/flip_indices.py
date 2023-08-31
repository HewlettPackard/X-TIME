"""
This module contains the `flip_indices` utility function CUDA kernel and public API.
"""

from string import Template

import cupy as cp
from numpy.typing import DTypeLike, NDArray

from ..types import IntDType, NumericDType, dtype_to_ctype, is_float_type
from .helpers import simple_kernel_dimensions

# Kernel notes:
#
# Dimensions:
#    Linear dimensions, i.e. only `threadIdx.x` and `blockIdx.x`.
#
# Operation per thread:
#    Each thread applies a single index in `indices`.

KERNEL_SOURCE_TEMPLATE = Template(
    r"""
extern "C" __global__
void flip_indices(
    $inputs_type *inputs, $indices_type *indices,
    long input_rows, long input_cols, long index_rows, long index_cols
) {
    long thread_id = threadIdx.x + (blockIdx.x * blockDim.x);

    if (thread_id >= input_rows * index_cols) {
        return;
    }

    $indices_type col_index = indices[thread_id % (index_rows * index_cols)];

    // ignore sub-zero indices
    if (col_index < 0) {
        return;
    }

    long row = thread_id / index_cols;
    inputs[row * input_cols + col_index] = 1 - inputs[row * input_cols + col_index];
}
"""
)


def generate_kernel(inputs_dtype: DTypeLike, indices_dtype: DTypeLike):
    """Converts the generic source template into an instance of `cp.RawKernel`"""

    if is_float_type(indices_dtype):
        raise TypeError(f"data type of indices is not an integer ({indices_dtype})")

    code = KERNEL_SOURCE_TEMPLATE.safe_substitute(
        inputs_type=dtype_to_ctype(inputs_dtype),
        indices_type=dtype_to_ctype(indices_dtype),
    )

    return cp.RawKernel(code=code, name="flip_indices")


def flip_indices(inputs: NDArray[NumericDType], indices: NDArray[IntDType]) -> None:
    """
    Flips (`x = 1 - x`) the elements in a two-dimensional input array indexed
    by a two-dimensional array of indices.

    ### Arguments:
    - `inputs` (`cupy.ndarray`): The two-dimensional input array that is
      mutated in-place.
    - `indices` (`numpy.ndarray` or `cupy.ndarray`): The two-dimensional indices.
      All indices less than zero are ignored.

    ### Returns:
    `None`

    ### Notes:
    - This is an in-place operation, so `inputs` is expected to be on the GPU.
    - `indices` can have less rows than `inputs`.
      In that case, `indices` is repeated across the rows of `inputs` such that they
      match.
    """

    if not isinstance(inputs, cp.ndarray):
        raise TypeError("expected inputs to be a cupy array already")

    if len(inputs.shape) < 2:
        raise ValueError("expected inputs to be two-dimensional")

    if len(indices.shape) < 2:
        raise ValueError("expected indices to be two-dimensional")

    index_rows, index_cols = indices.shape[-2], indices.shape[-1]
    input_rows, input_cols = inputs.shape[-2], inputs.shape[-1]

    if input_rows % index_rows != 0:
        raise ValueError(
            f"{index_rows} index rows can not evenly be repeated over {input_rows} input rows"  # noqa: E501
        )

    kernel = generate_kernel(inputs.dtype, indices.dtype)

    dims = simple_kernel_dimensions(
        input_rows * index_cols,  # one thread per index in the (repeated) indices array
        kernel.attributes["max_threads_per_block"],
    )

    indices = cp.asarray(indices.ravel())  # move `indices` to the GPU if needed

    kernel(*dims, (inputs, indices, input_rows, input_cols, index_rows, index_cols))
