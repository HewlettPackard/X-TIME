from typing import Optional

from numpy.typing import NDArray

from .types import CamOp, CamVariant, is_float_type


def validate_args(
    variant: CamVariant,
    op: CamOp,
    inputs: NDArray,
    cam: NDArray,
    reduction_values: Optional[NDArray] = None,
) -> None:
    """
    Ensures that the arguments for a CAM operation are sane and compatible
    by inspecting their dtypes and shapes.
    Raises appropriate errors in case a problem is found.
    """

    # dtype checking

    if inputs.dtype != cam.dtype:
        raise TypeError(
            f"inputs and cam dtypes are not equal: ({inputs.dtype} vs {cam.dtype})"
        )

    if op.is_reduction:
        if reduction_values is None:
            raise TypeError("operation is reduction but no reduction values were given")

        if not is_float_type(reduction_values.dtype):
            raise TypeError("reductions only support float data types")

    # shape checking

    if len(inputs.shape) < 2:
        raise ValueError("inputs are one-dimensional")

    if len(cam.shape) < 2:
        raise ValueError("cam is one-dimensional")

    cell_encoding_width = variant.cell_encoding_width

    if inputs.shape[-1] != cam.shape[-1] // cell_encoding_width:
        raise ValueError(
            f"input and cam column sizes are not equal ({inputs.shape[-1]} vs {cam.shape[-1] // cell_encoding_width})"  # noqa: E501
        )

    if cam.shape[-1] % cell_encoding_width != 0:
        raise ValueError(
            f"amount of columns not divisible by cell encoding width {cell_encoding_width} (columns: {cam.shape[-1]})"  # noqa: E501
        )

    if op.is_reduction:
        assert reduction_values is not None
        if len(reduction_values.shape) > 1:
            raise ValueError("reduction values must be one-dimensional")
        if len(inputs.shape) > 2 or len(cam.shape) > 2:
            raise ValueError("reduction operations do not support broadcasting")
        if reduction_values.shape != cam.shape[:-1]:
            raise ValueError("CAM shape does not match reduction value shape")

    inputs_outer_shape, cam_outer_shape = inputs.shape[:-2], cam.shape[:-2]

    for i in range(1, min(len(inputs_outer_shape), len(cam_outer_shape)) + 1):
        if inputs_outer_shape[-i] != cam_outer_shape[-i]:
            raise ValueError(
                f"the outer shapes of input and cam are incompatible: {inputs_outer_shape} vs {cam_outer_shape}"  # noqa: E501
            )
