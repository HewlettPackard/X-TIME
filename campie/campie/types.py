from enum import Enum, auto
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray


class CamVariant(Enum):
    """All supported variants of CAMs."""

    ACAM = auto()
    """Analog CAM."""

    TCAM = auto()
    """Ternary CAM."""

    @property
    def cell_encoding_width(self) -> int:
        """
        The amount of elements in a CAM row that is used
        to encode a single CAM cell.
        """

        if self == CamVariant.ACAM:
            return 2

        return 1


class CamOp(Enum):
    """All supported operations on CAMs."""

    MATCH = auto()
    """Get the match value for each row."""

    COUNT_MISMATCHES = auto()
    """
    Count the number of mismatches per row.
    For TCAMs, this is also called the hamming distance.
    """

    REDUCE_SUM = auto()
    """
    Get the match value per row and further reduce each input
    down to a single value, picked from a vector of values
    adjacent to the CAM.
    """

    @property
    def is_reduction(self) -> bool:
        """Whether the op is a reduction."""
        return self == CamOp.REDUCE_SUM

    def result_dtype(self, reduction_values: Optional[NDArray] = None) -> DTypeLike:
        """
        The resulting dtype of the op.
        For a reduction, this depends on the values that are reduced over.
        """

        if self == CamOp.MATCH:
            return np.int8

        elif self == CamOp.COUNT_MISMATCHES:
            return np.int64

        elif self == CamOp.REDUCE_SUM:
            if reduction_values is None:
                raise ValueError(
                    "could not determine resulting data type for CAM operation"
                )
            return reduction_values.dtype


FloatDType = Union[np.float32, np.float64]
"""All float NumPy data types."""

IntDType = Union[
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]
"""All integer NumPy data types."""

NumericDType = Union[IntDType, FloatDType]
"""All numeric NumPy data types, i.e., floats and ints."""

Dimensions = Tuple[int, int, int]
"""A 3-dimensional vector used for CUDA launch configurations."""

LaunchConfiguration = Tuple[Dimensions, Dimensions]
"""A CUDA kernel launch configuration in `(dim_grid, dim_block)` format."""

Kernel = Callable[[Dimensions, Dimensions, Tuple[Any, ...]], None]
"""A `cupy.RawKernel` instance."""

DTYPE_TO_CTYPE = {
    np.float32: "float",
    np.float64: "double",
    np.int8: "char",
    np.int16: "short",
    np.int32: "int",
    np.int64: "long",
    np.uint8: "char",
    np.uint16: "unsigned short",
    np.uint32: "unsigned int",
    np.uint64: "unsigned long",
}
"""Lookup table for numpy dtypes -> their type name in C."""


def dtype_to_ctype(dtype: DTypeLike) -> str:
    """Converts a numpy dtype to the name of its counterpart in C."""
    # directly indexing `_DTYPE_TO_CTYPE` doesn't work with dtype instances,
    # such as those obtained via `arr.dtype`. They need to be directly compared.
    for nptype in DTYPE_TO_CTYPE:
        if dtype == nptype:
            return DTYPE_TO_CTYPE[nptype]

    raise TypeError(f"data type {dtype} is not supported")


def is_float_type(dtype: DTypeLike) -> bool:
    """Determines whether a given data type is a floating point type."""
    return dtype == np.float32 or dtype == np.float64
