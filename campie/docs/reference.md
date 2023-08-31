# CAMPIE API reference

## Contents

- [campie.cam](#campiecam)
  - [campie.acam_count_mismatches](#campieacam_count_mismatches)
  - [campie.acam_match](#campieacam_match)
  - [campie.acam_reduce_sum](#campieacam_reduce_sum)
  - [campie.tcam_hamming_distance](#campietcam_hamming_distance)
  - [campie.tcam_match](#campietcam_match)
  - [campie.tcam_reduce_sum](#campietcam_reduce_sum)
- [campie.util](#campieutil)
  - [campie.flip_indices](#campieflip_indices)


# `campie.cam`

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


## `campie.acam_count_mismatches`

```python
def acam_count_mismatches(inputs, cam, noise=None)
```

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


## `campie.acam_match`

```python
def acam_match(inputs, cam, noise=None)
```

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


## `campie.acam_reduce_sum`

```python
def acam_reduce_sum(inputs, cam, values, noise=None)
```

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


## `campie.tcam_hamming_distance`

```python
def tcam_hamming_distance(inputs, cam)
```

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


## `campie.tcam_match`

```python
def tcam_match(inputs, cam)
```

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


## `campie.tcam_reduce_sum`

```python
def tcam_reduce_sum(inputs, cam, values)
```

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


# `campie.util`

This module includes additional utility functionality related to CAMs,
implemented as CUDA kernels.


## `campie.flip_indices`

```python
def flip_indices(inputs, indices)
```

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