"""Code generation for CUDA kernels that handle CAM operations."""

from string import Template
from typing import Dict, Tuple

import cupy as cp
from numpy.typing import DTypeLike

from .types import CamOp, CamVariant, Kernel, dtype_to_ctype, is_float_type

# Kernel notes:
#
# Dimensions:
#     The indexing inside of a two-dimensional pair of an input matrix and a CAM
#     is done linearly via `threadIdx.x` and `blockIdx.x`.
#     The indexing within the (flattened) stack of input and CAM pairs is done
#     via `blockIdx.y`.
#
# Operation per thread:
#     Each thread reduces an input/CAM column down to a single match.
#
# Shapes:
#     The kernel by itself only matches inputs and CAMs of the same dimensions,
#     including dimensions larger than two. To use this kernel with uneven dimensions,
#     it needs to be wrapped in additional code that flattens the uneven arguments
#     down to even ones and then reshapes the result back into the demanded shape.

KERNEL_OUTLINE = Template(
    r"""
extern "C" __global__
void cam(
    $inputs_type *inputs, $cam_type *cam,
    long columns, long input_rows, long cam_rows
    $extra_params
) {
    long block_id = blockIdx.x + blockIdx.y * gridDim.x;
    long thread_id = block_id * (blockDim.x * blockDim.y) + threadIdx.x;

    /* `gridDim.y` encodes the amount of cores in the stack.
        Return here in case the thread is not needed. */
    if (thread_id >= input_rows * cam_rows * gridDim.y) {
        return;
    }

    /* these are absolute indices, i.e., within the entire stack of inputs */
    long cam_row_index = blockIdx.y * cam_rows + thread_id % cam_rows;
    long input_row_index = thread_id / cam_rows;

    $pre_loop

    for (long i = 0; i < columns; i++) {
        $loop_contents
    }

    $post_loop
}
"""
)

TCAM_INT_MATCHING = Template(
    r"""
        $cam_type cam_value = cam[cam_row_index * columns + i];
        $inputs_type input_value = inputs[input_row_index * columns + i];

        if (cam_value > 1 || cam_value < 0) {
            continue;
        }

        if (input_value - cam_value != 0) {
            $on_mismatch
        }
"""
)

TCAM_FLOAT_MATCHING = Template(
    r"""
        $cam_type cam_value = cam[cam_row_index * columns + i];
        $inputs_type input_value = inputs[input_row_index * columns + i];

        if (fabs(input_value - cam_value) > 0.00001) {
            $on_mismatch
        }
"""
)

ACAM_FLOAT_MATCHING = Template(
    r"""
        $cam_type min_threshold = cam[2 * (cam_row_index * columns + i)];
        $cam_type max_threshold = cam[2 * (cam_row_index * columns + i) + 1];
        $inputs_type input_value = inputs[input_row_index * columns + i];

        /* nan = don't care */
        if (!(
            (isnan(min_threshold) || min_threshold <= input_value) &&
            (isnan(max_threshold) || input_value < max_threshold)
        )) {
            $on_mismatch
        }
"""
)

ACAM_INT_MATCHING = Template(
    r"""
        $cam_type min_threshold = cam[2 * (cam_row_index * columns + i)];
        $cam_type max_threshold = cam[2 * (cam_row_index * columns + i) + 1];
        $inputs_type input_value = inputs[input_row_index * columns + i];

        /* < 0 = don't care */
        if (!(
            (min_threshold < 0 || min_threshold <= input_value) &&
            (max_threshold < 0 || input_value < max_threshold)
        )) {
            $on_mismatch
        }
"""
)


def generate_kernel(
    variant: CamVariant,
    op: CamOp,
    inputs_dtype: DTypeLike,
    cam_dtype: DTypeLike,
    results_dtype: DTypeLike,
) -> Kernel:
    """Generates CUDA kernel code and returns a callable CAM kernel."""

    inputs_type = dtype_to_ctype(inputs_dtype)
    cam_type = dtype_to_ctype(cam_dtype)
    results_type = dtype_to_ctype(results_dtype)

    MATCHING_KINDS: Dict[Tuple[CamVariant, bool], Template] = {
        (CamVariant.TCAM, True): TCAM_FLOAT_MATCHING,
        (CamVariant.TCAM, False): TCAM_INT_MATCHING,
        (CamVariant.ACAM, True): ACAM_FLOAT_MATCHING,
        (CamVariant.ACAM, False): ACAM_INT_MATCHING,
    }

    matching = MATCHING_KINDS[(variant, is_float_type(cam_dtype))]

    if op == CamOp.MATCH:
        extra_params = f", {results_type} *matches"
        pre_loop = f"{results_type} match = 1;"
        post_loop = "matches[thread_id] = match;"
        on_mismatch = "match = 0; break;"

    elif op == CamOp.COUNT_MISMATCHES:
        extra_params = f", {results_type} *counts"
        pre_loop = f"{results_type} count = 0;"
        post_loop = "counts[thread_id] = count;"
        on_mismatch = "count++;"

    elif op == CamOp.REDUCE_SUM:
        extra_params = f", {results_type} *results, {results_type} *values"
        pre_loop = ""
        post_loop = "atomicAdd(&results[input_row_index], values[cam_row_index]);"
        on_mismatch = "return;"

    else:
        raise TypeError(f"unknown CAM operation: {op}")

    loop_contents = matching.safe_substitute(
        inputs_type=inputs_type,
        cam_type=cam_type,
        on_mismatch=on_mismatch,
    )

    code = KERNEL_OUTLINE.safe_substitute(
        inputs_type=inputs_type,
        cam_type=cam_type,
        results_type=results_type,
        extra_params=extra_params,
        loop_contents=loop_contents,
        pre_loop=pre_loop,
        post_loop=post_loop,
    )

    return cp.RawKernel(code=code, name="cam")
