# X-TIME Compiler

The compiler is exposed both as a command line interface and a Python library that binds to its native core written in Rust. Supported libraries are Scikit-Learn, XGBoost, CatBoost and LightGBM.

- [Building the CLI](#building-the-cli)
- [Building the Python Library](#building-the-python-library)
- [Usage in Python](#usage-in-python)

## Building the CLI

Install Rust:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  # select "default" (just press enter)
source ~/.cargo/env
```

Build the binary:

```sh
cargo build --release
```

This will place the `xtimec` binary inside the `target/release` folder

To automatically place it in your `$PATH`, you can alternatively use `cargo install`:

```sh
cargo install --path .
```

Usage of the command is as follows:

```
Usage: xtimec [OPTIONS] <INPUT_FILE> <OUTPUT_FILE>

Arguments:
  <INPUT_FILE>   Path to the model JSON dump
  <OUTPUT_FILE>  Path to the resulting NumPy file

Options:
  -t, --type <MODEL_TYPE>  The type of model that is passed [default: treelite] [possible values: treelite, catboost]
  -h, --help               Print help
  -V, --version            Print version
```

## Building the Python Library

Install Rust:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  # select "default" (just press enter)
source ~/.cargo/env
```

You should now have the `cargo` command available.

Create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Install Maturin:

```sh
pip install maturin
```

Build the project:

```sh
maturin build --release
```

This will place a `.whl` file that you can `pip install` inside of the `target/wheels` folder.

Alternatively, you can use:

```sh
maturin develop
```

to enter a development shell with the library and all its dependencies available.

## Usage in Python

Example:

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from xtimec import XTimeModel

# create a random dataset
X, y = make_classification(n_samples=100, n_informative=5, n_classes=2)

# train a model
model = RandomForestClassifier()
model.fit(X, y)

# represent the model encoded for an analog CAM
# alternatively `from_xgboost`, `from_catboost` etc.
xmodel = XTimeModel.from_sklearn(model)

# >>> xmodel.cam
#   np.array([[...]])
# >>> xmodel.leaves
#   np.array([[...]])
```

See [`python/xtimec/model.py`](./python/xtimec/model.py) for the entire Python API.
