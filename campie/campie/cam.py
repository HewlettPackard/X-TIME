"""
This module contains all CAM related functions.
Below are some general notes that apply to all functions in this module.

### Notes:
- Both the CAM and its inputs are always expected to contain the same data type.
- NumPy arrays passed to functions in this module will automatically be moved to
  the GPU. If you are calling a function multiple times with a fixed subset of
  parameters, consider moving them to the GPU beforehand yourself to avoid
  unnecessary data transfer between the host and the GPU. You can move NumPy arrays
  to the GPU via `cupy.asarray`.
- The arrays returned by functions in this module are always still on the GPU.
  To retrieve them to the host, call `.get()` or `cupy.asnumpy(...)`.

### Broadcasting:
- If both the inputs and the CAM are 2-D, they are matched like a conventional
  set of inputs and CAM.
- If both the inputs and the CAM are N-D, N > 2, they are treated like a stack of
  inputs and CAMs and matching only takes place within the last two dimensions.
- If the inputs are N-D, N >= 2 and the CAMs are M-D, M >= 2, the smaller
  argument is repeatedly applied to the greater one.
- This is analogous to the way that matrix multiplication works in NumPy
  with dimensions larger than 2.
"""

from typing import Optional, TypeVar

import cupy as cp
import numpy as np
from numpy.typing import NDArray

from .kernel import generate_kernel
from .run import run_kernel
from .types import CamOp, CamVariant, FloatDType, NumericDType
from .validation import validate_args

TN = TypeVar("TN", bound=NumericDType)
TF = TypeVar("TF", bound=FloatDType)
UF = TypeVar("UF", bound=FloatDType)


def tcam_match(inputs: NDArray[TN], cam: NDArray[TN]) -> NDArray[np.int8]:
    """
    Performs the matching operation of a TCAM (Ternary CAM) on a given set of inputs and
    TCAMs.

    ### Arguments:
    - `inputs` (`numpy.ndarray` or `cupy.ndarray`): The input columns stacked in rows.
      The two innermost dimensions of `inputs` are of shape `input_rows x columns`.
    - `cam` (`numpy.ndarray` or `cupy.ndarray`): The TCAM matrix itself.
      When using a float dytpe, a "Don't Care" (or "X") is encoded as `numpy.nan`.
      Alternatively, for integer dtypes they are indicated by values < 0 or > 1.
      The two innermost dimensions of `cam` are of shape `cam_rows x columns`.

    ### Returns:
    `cupy.ndarray`: The resulting matrix of matches encoded as `np.int8`s.
    The two innermost dimensions are of shape `input_rows x cam_rows`.
    """
    variant, op, result_dtype = CamVariant.TCAM, CamOp.MATCH, np.int8
    validate_args(variant, op, inputs, cam)
    kernel = generate_kernel(variant, op, inputs.dtype, cam.dtype, result_dtype)
    return run_kernel(kernel, variant, op, inputs, cam, result_dtype)


def tcam_hamming_distance(inputs: NDArray[TN], cam: NDArray[TN]) -> NDArray[np.int64]:
    """
    Determines the hamming distance (number of mismatches per row) in a
    TCAM (Ternary CAM) matching operation for a given set of inputs and TCAMs.

    ### Arguments:
    - `inputs` (`numpy.ndarray` or `cupy.ndarray`): The input columns stacked in rows.
      The two innermost dimensions of `inputs` are of shape `input_rows x columns`.
    - `cam` (`numpy.ndarray` or `cupy.ndarray`): The TCAM matrix itself.
      When using a float dytpe, a "Don't Care" (or "X") is encoded as `numpy.nan`.
      Alternatively, for integer dtypes they are indicated by values < 0 or > 1.
      The two innermost dimensions of `cam` are of shape `cam_rows x columns`.

    ### Returns:
    `cupy.ndarray`: The resulting matrix of mismatch counts encoded as `np.int64`s.
    The two innermost dimensions are of shape `input_rows x cam_rows`.
    """
    variant, op, result_dtype = CamVariant.TCAM, CamOp.COUNT_MISMATCHES, np.int64
    validate_args(variant, op, inputs, cam)
    kernel = generate_kernel(variant, op, inputs.dtype, cam.dtype, result_dtype)
    return run_kernel(kernel, variant, op, inputs, cam, result_dtype)


def tcam_reduce_sum(
    inputs: NDArray[TN], cam: NDArray[TN], *, values: NDArray[TF]
) -> NDArray[TF]:
    """
    First performs the matching operation of a TCAM (Ternary CAM) on a given set
    of inputs and TCAMs, then reduces each row of inputs down to a single number
    by accumulating over a set of values for every matched row.

    Note that reduction operations do not support broadcasting.

    ### Positional Arguments:
    - `inputs` (`numpy.ndarray` or `cupy.ndarray`): The input columns stacked in rows.
      The two innermost dimensions of `inputs` are of shape `input_rows x columns`.
    - `cam` (`numpy.ndarray` or `cupy.ndarray`): The TCAM matrix itself.
      When using a float dytpe, a "Don't Care" (or "X") is encoded as `numpy.nan`.
      Alternatively, for integer dtypes they are indicated by values < 0 or > 1.
      The two innermost dimensions of `cam` are of shape `cam_rows x columns`.

    ### Keyword Arguments:
    - `values` (`numpy.ndarray` or `cupy.ndarray`): The values to reduce.
      The dimensions of `values` must be the same as the dimensions of `cam`,
      except for the omission of the `columns` dimension.

    ### Returns:
    `cupy.ndarray`: The reduction result matrix.
    The innermost dimension is `input_rows`.
    """
    variant, op, result_dtype = CamVariant.TCAM, CamOp.REDUCE_SUM, values.dtype
    validate_args(variant, op, inputs, cam, values)
    kernel = generate_kernel(variant, op, inputs.dtype, cam.dtype, result_dtype)
    return run_kernel(kernel, variant, op, inputs, cam, result_dtype, values)


def add_noise(acam: NDArray, noise: float) -> NDArray:
    """Adds noise to ACAM thresholds based on a normal distribution."""
    xp = cp.get_array_module(acam)
    return acam + xp.random.normal(0, noise, acam.shape).astype(acam.dtype)


def acam_match(
    inputs: NDArray[TN], cam: NDArray[TN], *, noise: Optional[float] = None
) -> NDArray[np.int8]:
    """
    Performs the matching operation of a ACAM (Analog CAM) on a given set of inputs and
    ACAMs.

    ### Positional Arguments:
    - `inputs` (`numpy.ndarray` or `cupy.ndarray`): The input columns stacked in rows.
      The two innermost dimensions of `inputs` are of shape `input_rows x columns`.
    - `cam` (`numpy.ndarray` or `cupy.ndarray`): The ACAM matrix itself.
      The lower and upper thresholds are encoded side-by-side within the column.
      This means that your CAM columns should be twice as wide as your input columns.
      "Don't Care" (or "X") thresholds are encoded as `numpy.nan` values when using
      float types and as negative integers when using integer types.
      The two innermost dimensions of `cam` are of shape `cam_rows x (columns * 2)`.

    ### Keyword Arguments:
    - `noise` (`float` [optional]): Standard deviation for a normal distribution
      `N(0, noise)` that is randomly sampled from and added to the ACAM thresholds
      to simulate analog inaccuracies before performing the operation. Noise is only
      added if this argument is defined.

    ### Returns:
    `cupy.ndarray`: The resulting matrix of matches encoded as `np.int8`s.
    The two innermost dimensions are of shape `input_rows x cam_rows`.

    ### Notes:
    - The arguments for ACAM operations must contain floating point data types.
    """
    variant, op, result_dtype = CamVariant.ACAM, CamOp.MATCH, np.int8
    validate_args(variant, op, inputs, cam)
    kernel = generate_kernel(variant, op, inputs.dtype, cam.dtype, result_dtype)
    return run_kernel(kernel, variant, op, inputs, cam, result_dtype)


def acam_count_mismatches(
    inputs: NDArray[TN], cam: NDArray[TN], *, noise: Optional[float] = None
) -> NDArray[np.int64]:
    """
    Determines the number of mismatches per row in a ACAM (Analog CAM) matching
    operation for a given set of inputs and ACAMs.

    ### Positional Arguments:
    - `inputs` (`numpy.ndarray` or `cupy.ndarray`): The input columns stacked in rows.
      The two innermost dimensions of `inputs` are of shape `input_rows x columns`.
    - `cam` (`numpy.ndarray` or `cupy.ndarray`): The ACAM matrix itself.
      The lower and upper thresholds are encoded side-by-side within the column.
      This means that your CAM columns should be twice as wide as your input columns.
      "Don't Care" (or "X") thresholds are encoded as `numpy.nan` values when using
      float types and as negative integers when using integer types.
      The two innermost dimensions of `cam` are of shape `cam_rows x (columns * 2)`.

    ### Keyword Arguments:
    - `noise` (`float` [optional]): Standard deviation for a normal distribution
      `N(0, noise)` that is randomly sampled from and added to the ACAM thresholds
      to simulate analog inaccuracies before performing the operation. Noise is only
      added if this argument is defined.

    ### Returns:
    `cupy.ndarray`: The resulting matrix of mismatch counts encoded as `np.int64`s.
    The two innermost dimensions are of shape `input_rows x cam_rows`.

    ### Notes:
    - The arguments for ACAM operations must contain floating point data types.
    """
    variant, op, result_dtype = CamVariant.ACAM, CamOp.COUNT_MISMATCHES, np.int64
    validate_args(variant, op, inputs, cam)
    kernel = generate_kernel(variant, op, inputs.dtype, cam.dtype, result_dtype)
    if noise is not None:
        cam = add_noise(cam, noise)
    return run_kernel(kernel, variant, op, inputs, cam, result_dtype)


def acam_reduce_sum(
    inputs: NDArray[TN],
    cam: NDArray[TN],
    *,
    values: NDArray[UF],
    noise: Optional[float] = None
) -> NDArray[UF]:
    """
    First performs the matching operation of a ACAM (Analog CAM) on a given set
    of inputs and ACAMs, then reduces each row of inputs down to a single number
    by accumulating over a set of values for every matched row.

    Note that reduction operations do not support broadcasting.

    ### Positional Arguments:
    - `inputs` (`numpy.ndarray` or `cupy.ndarray`): The input columns stacked in rows.
      The two innermost dimensions of `inputs` are of shape `input_rows x columns`.
    - `cam` (`numpy.ndarray` or `cupy.ndarray`): The ACAM matrix itself.
      The lower and upper thresholds are encoded side-by-side within the column.
      This means that your CAM columns should be twice as wide as your input columns.
      "Don't Care" (or "X") thresholds are encoded as `numpy.nan` values when using
      float types and as negative integers when using integer types.
      The two innermost dimensions of `cam` are of shape `cam_rows x (columns * 2)`.

    ### Keyword Arguments:
    - `values` (`numpy.ndarray` or `cupy.ndarray`): The values to reduce.
      The dimensions of `values` must be the same as the dimensions of `cam`,
      except for the omission of the `columns` dimension.
    - `noise` (`float` [optional]): Standard deviation for a normal distribution
      `N(0, noise)` that is randomly sampled from and added to the ACAM thresholds
      to simulate analog inaccuracies before performing the operation. Noise is only
      added if this argument is defined.

    ### Returns:
    `cupy.ndarray`: The reduction result matrix.
    The innermost dimension is `input_rows`.

    ### Notes:
    - The arguments for ACAM operations must contain floating point data types.
    """
    variant, op, result_dtype = CamVariant.ACAM, CamOp.REDUCE_SUM, values.dtype
    validate_args(variant, op, inputs, cam, values)
    kernel = generate_kernel(variant, op, inputs.dtype, cam.dtype, result_dtype)
    if noise is not None:
        cam = add_noise(cam, noise)
    return run_kernel(kernel, variant, op, inputs, cam, result_dtype, values)
