# CAMPIE 🥧

Python APIs to simulate various CAMs
([Content Addressable Memory]("https://en.wikipedia.org/wiki/Content-addressable_memory"))
on GPUs at scale.

-   [Installation](#installation)
-   [Example](#example)
-   [Documentation](./docs/reference.md)
-   [Examples](./docs/example.ipynb)
-   [Contributing](#contributing)

## Overview

CAMPIE is a utility library that builds on top of [NumPy](https://numpy.org) and
[CuPy](https://cupy.dev) for fast and efficient simulation of CAM hardwares on CUDA GPUs. \
It implements custom CUDA kernels for ACAM (Analog CAM) and TCAM (Ternary Cam) simulation
and adjacent utility functionality accessible under a simple Python API.

## Installation

CAMPIE uses [CuPy](https://cupy.dev) under the hood,
which requires you to install a different package based on your CUDA version if you are using pip.

The CUDA version that you are installing CAMPIE for is specified as an [extra](https://peps.python.org/pep-0508/#extras) like below:

```
pip install campie[cu11x]
```

for [Poetry](https://github.com/python-poetry/poetry), this would be:

```sh
poetry add -E cu11x campie
```

The available CUDA version extras are as follows: `cu110`, `cu111`, `cu11x`, `cu12x`.
You should only ever install one CUDA extra or you will cause conflicts.

For more context, see the [CuPy installation instructions](https://github.com/cupy/cupy#pip) and
the [`pyproject.toml`](./pyproject.toml).

## Example

```python
import campie
import numpy as np

x = np.nan  # nan = don't care

# cam_rows x columns
cam = np.array([
  [0, 0, 1, 0],
  [1, 1, x, 0],
  [0, 0, 0, 0],
  [x, x, 0, 0],
  [0, 0, 1, 1],
])

# input_rows x columns
inputs = np.array([
  [0, 0, 0, 0],
  [0, 1, 0, 0],
  [1, 1, 1, 0],
]).astype(np.float64)

# this runs on the GPU, `matches` is still in GPU memory
matches = campie.tcam_match(inputs, cam)
print(matches)

# -> input_rows x cam_rows
array([
  [0, 0, 1, 1, 0],  # input 1 matches cam rows 3 and 4
  [0, 0, 0, 1, 0],  # input 2 matches cam row 4
  [0, 1, 0, 0, 0],  # input 3 matches cam row 2
])
```

For detailed information on all available APIs, visit the [documentation](./docs/reference.md).
Alternatively, see the [example notebook](./docs/example.ipynb) for a practical introduction.

## Contributing

### Dependency Management

Python dependencies are managed via [Poetry](https://github.com/python-poetry/poetry):

```sh
poetry install -E cu11x
```

Poetry is also used to publish the library to PyPI:

```sh
poetry build
poetry publish
```

### Formatting & Linting

CAMPIE uses [black](https://github.com/psf/black) and [isort](https://github.com/PyCQA/isort)
to format Python source code.

To format all Python files:

```sh
black .
isort .
```

Additionally, CAMPIE uses [ruff](https://ruff.rs) to lint Python source code.

To lint all Python files:

```sh
ruff .
```

Finally, you can also use the script provided in [`scripts/format.py`](./scripts/format.py) to
format and lint everything at once.

```sh
# installed into the virtual environment by poetry
format

# or alternatively
python scripts/format.py
```

### Reference Generation

[`scripts/gen_reference.py`](./docs/gen_reference.py) is used to extract doc comments from the source
code and generates [`docs/reference.md`](./docs/reference.md).

To regenerate the reference:

```sh
# installed into the virtual environment by poetry
gen-reference

# or alternatively
python scripts/gen_reference.py
```
