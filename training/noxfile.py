###
# Copyright (2023) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import inspect
import os
import typing as t
from pathlib import Path

import nox
import tomlkit

XTIME_NOX_PYTHON_VERSIONS = ["3.9", "3.10", "3.11"]
"""The list of python versions to run nox sessions with. Can be overridden by setting the environment variable."""
if "XTIME_NOX_PYTHON_VERSIONS" in os.environ:
    XTIME_NOX_PYTHON_VERSIONS = os.environ["XTIME_NOX_PYTHON_VERSIONS"].split(",")


# Prevent Python from writing bytecode
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


@nox.session(python=XTIME_NOX_PYTHON_VERSIONS, name="unit")
@nox.parametrize("deps", ["pinned", "latest"])
def unit_tests(session: nox.Session, deps: str) -> None:
    """Run XTIME training unit tests.

    The `posargs` are passed through to pytest (e.g., -m datasets). Current working directory - location of the
    `noxfile.py`. On a command line, these additional args should follow after `--`. For example:
    ```
    nox -s unit-3.11(deps='pinned') -- -n 8
    ```
    where:
        `-n NUM`: number of parallel unit tests to run (with `pytest-xdist`)

    Args:
        session: Current nox session.
        deps: When pinned, all dependencies are fixed to those specified in `poetry.lock`. When latest, only primary
            dependencies specified in the pyproject.toml may be pinned. All other, secondary dependencies, may or may
            not be pinned depending on metadata of primary dependencies. When pinned dependencies are used, the poetry
            must be available externally.
    """
    # Install this project and pytest (with `pip install`).
    install_args: t.List[str] = [".[all]", "pytest", "pytest-xdist"]

    # If pinned deps to be used, export constraints from `poetry.lock` file using `poetry`.
    if deps == "pinned":
        constraints_file: str = (Path(session.create_tmp()).resolve() / "constraints.txt").as_posix()
        session.run(
            "poetry",
            "export",
            "--format=constraints.txt",
            f"--output={constraints_file}",
            "--without-hashes",
            "--with=dev",
            "--without=eda",
            "--extras=all",
            external=True,
        )
        install_args.append(f"--constraint={constraints_file}")

    session.install(*install_args)
    session.run("pytest", "-v", *session.posargs)


@nox.session()
@nox.parametrize("model", ["xgboost", "catboost", "lightgbm"])
def model_train_test(session: nox.Session, model: str) -> None:
    """Train a model and validate the output of the training stage.

    Command to describe a dataset:
        `python -m xtime.main dataset describe churn_modelling:default`
    Command to run a training session:
        `python -m xtime.main experiment train --params="params:n_estimators=10;random_state=1" dataset model`.

    Args:
        session: Current nox session.
        model: Model to use. This must be a name for model's extra dependency (see
            pyproject.toml -> [tool.poetry.extras] table). It is assumed that this name is the same as model name that
            users provide on a command line (this is true for `catboost`, `xgboost` and `lightgbm`).
    """
    # Run the training session.
    dataset: str = "churn_modelling:default"  # We will be using this dataset
    session.install(f".[{model}]")  # Install only one extra dependency - xgboost
    # Just in case - make sure the dataset is available
    # session.run("python", "-m", "xtime.main", "dataset", "describe", dataset)
    # We will use file system based MLflow backend
    session_tmp_dir = Path(session.create_tmp()).resolve()

    mlflow_uri: Path = session_tmp_dir / ".mlruns"
    env_vars = {"MLFLOW_TRACKING_URI": mlflow_uri.as_uri(), "MLFLOW_EXPERIMENT_NAME": f"test_{model}"}

    session.run(
        "python",
        "-m",
        "xtime.main",
        "experiment",
        "train",
        "--params=params:n_estimators=100;random_state=1",
        dataset,
        model,
        env=env_vars,
    )

    # Validate output files
    validate_file = session_tmp_dir / "validate.py"
    with open(validate_file, "wt") as f:
        validate_train_run_source: str = inspect.getsource(_validate_train_run)
        f.write(validate_train_run_source)
        f.write("\n_validate_train_run()\n")

    session.run("python", validate_file.as_posix(), env=env_vars)


@nox.session()
@nox.parametrize("model", ["xgboost", "catboost", "lightgbm"])
def model_search_hp_test(session: nox.Session, model: str) -> None:
    """Run hyperparameter search stage and validate its output.

    Args:
        session: Current nox session.
        model: Model to use. This must be a name for model's extra dependency (see
            pyproject.toml -> [tool.poetry.extras] table). It is assumed that this name is the same as model name that
            users provide on a command line (this is true for `catboost`, `xgboost` and `lightgbm`).
    """
    # Run the training session.
    dataset: str = "churn_modelling:default"  # We will be using this dataset
    session.install(f".[{model}]")  # Install only one extra dependency - xgboost
    # Just in case - make sure the dataset is available
    session.run("python", "-m", "xtime.main", "dataset", "describe", dataset)
    # We will use file system based MLflow backend
    session_tmp_dir = Path(session.create_tmp()).resolve()

    mlflow_uri: Path = session_tmp_dir / ".mlruns"
    num_search_trials: int = 8
    env_vars = {
        "MLFLOW_TRACKING_URI": mlflow_uri.as_uri(),
        "MLFLOW_EXPERIMENT_NAME": f"test_{model}",
        "XTIME_NUM_SEARCH_TRIALS": str(num_search_trials),
        "XTIME_NOX_TEST": "1",
    }

    session.run(
        "python", "-m", "xtime.main", "experiment", "search_hp",
        f"--num-search-trials={num_search_trials}",
        "--num-validate-trials=0",
        "--params", f"auto:default:model={model};task=binary_classification",
        "--params", "params:n_estimators=10;random_state=1",
        dataset,
        model,
        "random",
        env=env_vars,
    )  # fmt: skip

    # Validate output files
    validate_file = _create_validate_file(session_tmp_dir, _validate_search_hp_run, [_validate_xtime_train_run])
    session.run("python", validate_file.as_posix(), env=env_vars)


@nox.session()
def test_pyproject_toml(session: nox.Session) -> None:
    """Run various checks on pyproject.toml."""
    pyproject = tomlkit.parse((Path(__file__).parent / "pyproject.toml").read_bytes().decode("utf-8"))
    version = pyproject["tool"]["poetry"]["version"]
    if version != "0.0.0":
        raise ValueError(f"Invalid project version ({version}). Expected value is '0.0.0'.")


def _validate_search_hp_run() -> None:
    """Validate `xtime.training` hyperparameter search run.

    The test driver will get the source of this function and will write it into a file. Then, it will execute it
    in the session python runtime.

    This function uses mlflow, ray tune and xtime API. It validates the presence of MLflow runs and output artifacts
    (and content of these artifacts to some extent). It validates the ray_tune directory, and also validates each trial
    folder.
    """
    import os
    import typing as t

    import mlflow
    from mlflow.entities import Run

    from xtime.contrib.mlflow_ext import MLflow
    from xtime.contrib.tune_ext import Analysis
    from xtime.io import IO

    experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]  # test_MODEL
    model_name = experiment_name[5:]
    num_search_trials: int = int(os.environ["XTIME_NUM_SEARCH_TRIALS"])

    # There must be one experiment (Default + this one).
    experiment_ids: t.List[str] = MLflow.get_experiment_ids()
    assert (
        len(experiment_ids) == 2
    ), f"Expected two experiments (Default and {experiment_name}), but got {len(experiment_ids)}."
    assert all(
        mlflow.get_experiment(eid).name in {"Default", experiment_name} for eid in experiment_ids
    ), "Unexpected experiment name (s)."

    # There must be one run
    runs: t.List[Run] = MLflow.get_runs()
    assert len(runs) == 1, f"Expected one run, but got {len(runs)}."

    # Verify run outputs
    run: Run = runs[0]

    artifact_path: Path = MLflow.get_artifact_path(run, ensure_exists=False, raise_if_not_exist=True)

    # Validate all files and directories exist
    expected_files = ["best_trial.yaml", "dataset_info.yaml", "run_inputs.yaml", "summary.yaml"]
    missing_files = [f for f in expected_files if not (artifact_path / f).is_file()]
    assert not missing_files, f"Missing files in the run artifact path: {missing_files}."
    assert (artifact_path / "ray_tune").is_dir(), f"Missing `ray_tune` directory in {artifact_path}."

    # Validate `best_trial.yaml` file.
    _ = IO.load_dict(
        artifact_path / "best_trial.yaml",
        expected_keys={
            "config", "local_path", "metrics", "num_failed_trials", "num_successful_trials", "relative_path",
            "run_uri", "trial_uri",
        },
    )  # fmt: skip

    # Validate `dataset_info.yaml` file.
    _ = IO.load_dict(
        artifact_path / "dataset_info.yaml", expected_keys={"features", "name", "properties", "task", "version"}
    )

    # Validate `run_inputs.yaml` file.
    run_inputs = IO.load_dict(artifact_path / "run_inputs.yaml", expected_keys={"dataset", "hparams", "model"})
    assert run_inputs["model"] == model_name, f"Invalid run inputs: {run_inputs}."

    # Validate `summary.yaml` file
    summary_from_file = IO.load_dict(
        artifact_path / "summary.yaml", expected_keys={"best_run", "metric_variations", "mlflow_run", "problem", "tags"}
    )

    # Validate ray tune trials using ray tune API
    trial_stats: t.List[t.Dict] = Analysis.get_trial_stats(run.info.run_id, skip_model_stats=True)
    assert len(trial_stats) == num_search_trials, "Number of trials do not match."

    summary_from_api: t.Dict = Analysis.get_summary(run.info.run_id)
    assert all(k in summary_from_file for k in summary_from_api.keys()), "Summary from API does not contain all keys."

    best_trial = Analysis.get_best_trial(run.info.run_id)
    assert best_trial, "Best trial is empty"

    # Validate individual trials
    for trial_stat in trial_stats:
        _validate_xtime_train_run(Path(trial_stat["tune_trial_path"]), model_name)


def _validate_train_run() -> None:
    """Validate `xtime.training` train run.

    The test driver will get the source of this function and will write it into a file. Then, it will execute it
    in the session python runtime. In current implementation, this function must be self-contained, e.g., keep all
    imports needed by this function in the function body.

    This function uses mlflow and xtime API. It validates the presence of MLflow runs and output artifacts (and content
    of these artifacts to some extent). This function does not use ray tune API.
    """
    import os
    import typing as t  # noqa

    import mlflow
    from mlflow.entities import Experiment, Run

    from xtime.contrib.mlflow_ext import MLflow
    from xtime.estimators.estimator import LegacySavedModelInfo, LegacySavedModelLoader
    from xtime.io import IO

    n = 10  # number of validation steps

    experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]  # test_MODEL
    model_name = experiment_name[5:]
    print(f"[validate_train_run] 00/{n} start validation experiment={experiment_name}, model={model_name}.")

    # There must be one experiment.
    experiment_ids: t.List[str] = MLflow.get_experiment_ids()
    if len(experiment_ids) != 2:
        raise ValueError(f"Expected two experiments (Default and {experiment_name}), but got {len(experiment_ids)}.")
    for experiment_id in experiment_ids:
        experiment: Experiment = mlflow.get_experiment(experiment_id)
        if experiment.name not in {"Default", experiment_name}:
            raise ValueError(f"Unexpected experiment name: {experiment.name}.")
    print(f"[validate_train_run] 01/{n} experiments validated experiment_ids={experiment_ids}.")

    # There must be one run
    runs: t.List[Run] = MLflow.get_runs()
    if len(runs) != 1:
        raise ValueError(f"Expected one run, but got {len(runs)}.")
    print(f"[validate_train_run] 02/{n} runs validated run_ids=[{runs[0].info.run_id}].")

    # Verify run outputs
    run: Run = runs[0]

    artifact_path: Path = MLflow.get_artifact_path(run, ensure_exists=False)
    if not artifact_path.is_dir():
        raise ValueError(f"Run artifact path ({artifact_path}) does not exist or not directory.")
    print(f"[validate_train_run] 03/{n} artifact path validated artifact_path={artifact_path.as_posix()}.")

    saved_model_info = LegacySavedModelInfo.from_path(artifact_path)
    expected_files = ["data_info.yaml", "model_info.yaml", "run_info.yaml", "run_inputs.yaml", "test_info.yaml"] + [
        saved_model_info.file_name()
    ]

    missing_files = [f for f in expected_files if not (artifact_path / f).is_file()]
    if missing_files:
        raise ValueError(f"Missing files in the run artifact path: {missing_files}.")
    print(f"[validate_train_run] 04/{n} files presence validated expected_files={expected_files}.")

    _ = IO.load_dict(
        artifact_path / "data_info.yaml", expected_keys={"features", "name", "properties", "task", "version"}
    )
    print(f"[validate_train_run] 05/{n} data_info.yaml validated.")

    model_info = IO.load_dict(artifact_path / "model_info.yaml", expected_keys={"model"})
    if model_info["model"]["name"] != model_name:
        raise ValueError(f"Invalid model info: {model_info}.")
    print(f"[validate_train_run] 06/{n} model_info.yaml validated.")

    _ = IO.load_dict(
        artifact_path / "run_info.yaml", expected_keys={"context", "env", "estimator", "hparams", "metadata"}
    )
    print(f"[validate_train_run] 07/{n} run_info.yaml validated.")

    run_inputs = IO.load_dict(artifact_path / "run_inputs.yaml", expected_keys={"dataset", "hparams", "model"})
    if run_inputs["model"] != model_name:
        raise ValueError(f"Invalid run inputs: {model_info}.")
    print(f"[validate_train_run] 08/{n} run_inputs.yaml validated.")

    _ = IO.load_dict(
        artifact_path / "test_info.yaml",
        expected_keys={
            "dataset_accuracy", "dataset_loss_mean", "dataset_loss_total", "test_accuracy", "test_auc", "test_f1",
            "test_loss_mean", "test_loss_total", "test_precision", "test_recall", "train_accuracy", "train_auc",
            "train_f1", "train_loss_mean", "train_loss_total", "train_precision", "train_recall", "valid_accuracy",
            "valid_loss_mean", "valid_loss_total"
        }
    )  # fmt: skip
    print(f"[validate_train_run] 09/{n} test_info.yaml validated.")

    model = LegacySavedModelLoader.load_model(artifact_path, saved_model_info)
    if model is None:
        raise ValueError("Failed to load the model.")
    print(f"[validate_train_run] 10/{n} model validated.")


def _validate_xtime_train_run(train_dir: Path, model_name: str) -> None:
    """Partially validate XTIME train run.

    Validates the following: data_info.yaml, model_info.yaml, run_info.yaml and test_info.yaml. Also validates
        model file.

    Args:
        train_dir: Training directory.
        model_name: Expected name of the model trained in this run.
    """
    from xtime.estimators.estimator import LegacySavedModelInfo, LegacySavedModelLoader
    from xtime.io import IO

    saved_model_info = LegacySavedModelInfo.from_path(train_dir)
    expected_files = ["data_info.yaml", "model_info.yaml", "run_info.yaml", "test_info.yaml"] + [
        saved_model_info.file_name()
    ]
    missing_files = [f for f in expected_files if not (train_dir / f).is_file()]
    assert not missing_files, f"Missing files in the run artifact path: {missing_files}."

    _ = IO.load_dict(train_dir / "data_info.yaml", expected_keys={"features", "name", "properties", "task", "version"})

    model_info = IO.load_dict(train_dir / "model_info.yaml", expected_keys={"model"})
    assert model_info["model"]["name"] == model_name, f"Invalid model info: {model_info}."

    _ = IO.load_dict(train_dir / "run_info.yaml", expected_keys={"context", "env", "estimator", "hparams", "metadata"})

    _ = IO.load_dict(
        train_dir / "test_info.yaml",
        expected_keys={
            "dataset_accuracy", "dataset_loss_mean", "dataset_loss_total", "test_accuracy", "test_auc", "test_f1",
            "test_loss_mean", "test_loss_total", "test_precision", "test_recall", "train_accuracy", "train_auc",
            "train_f1", "train_loss_mean", "train_loss_total", "train_precision", "train_recall", "valid_accuracy",
            "valid_loss_mean", "valid_loss_total"
        }
    )  # fmt: skip

    model = LegacySavedModelLoader.load_model(train_dir, saved_model_info)
    assert model is not None, "Failed to load the model."


def _create_validate_file(location: Path, main_fn: t.Callable, deps: t.List[t.Callable]) -> Path:
    """Create a `validate.py` file to run in a nox session to validate outputs if XTIME stages (train / search_hp).

    Args:
        location: Location to write file to.
        main_fn: Function to run in the validation file.
        deps: Other functions that `main_fn` calls. Will be written to validation file.

    Returns:
        Full path to `validate.py` file.
    """
    validate_file = location / "validate.py"
    with open(validate_file, "wt") as f:
        f.write("from pathlib import Path\n")  # for func annotations
        f.write(inspect.getsource(main_fn) + "\n")
        for dep in deps:
            f.write(inspect.getsource(dep) + "\n")
        f.write(f"{main_fn.__name__}()\n")
    return validate_file


# if __name__ == "__main__":
#    _validate_search_hp_run()
