# XTIME training unit tests.


1. Install `test` dependencies.
2. Run unit tests with `pytest`:
   ```shell
   # Run all unit tests including unit tests for datasets and estimators 
   # (ML models). See below to disable certain tests.
   python -m pytest -n auto
   ```

## Multiprocessing
XTIME uses pytest's `pytest-xdist` plugin to run tests in parallel. Multiprocessing is enabled by providing `-n auto`
command line argument to pytest.

## Unite test selection
Some unit tests (those related to datasets and ML models) require ML datasets to be available on the target system. If
these tests do not need to be run, disable them with `-m "not datasets and not estimators"`:
```shell
python -m pytest -n auto -m "not datasets and not estimators"
```
See [pyproject.toml](../pyproject.toml) project configuration file for all project-specific marks 
(`[tool.pytest.ini_options]` section).
