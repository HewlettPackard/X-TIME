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

import copy
import logging
import sys
import typing as t
from pathlib import Path

import mlflow
import ray
from mlflow import ActiveRun
from omegaconf import DictConfig, OmegaConf
from ray import tune
from ray.air import Result, RunConfig
from ray.tune import ResultGrid, TuneConfig
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch

import xtime.contrib.tune_ext as ray_tune_extensions
from xtime.contrib.mlflow_ext import MLflow
from xtime.contrib.tune_ext import Analysis, RayTuneDriverToMLflowLoggerCallback
from xtime.contrib.utils import Text, log_deprecate_msg_for_run_inputs
from xtime.datasets import build_dataset
from xtime.estimators import Estimator, get_estimator
from xtime.hparams import HParamsSource, default_hparams, get_hparams
from xtime.io import IO, encode
from xtime.ml import METRICS
from xtime.run import Context, Metadata, RunType

__all__ = ["run", "create_example_config"]


logger = logging.getLogger(__name__)


def run(config: DictConfig) -> None:
    config = OmegaConf.merge(_DEFAULT_CONFIGURATION, config)
    assert config.stage == "search_hp", f"Invalid stage {config.stage}."
    validation_config: DictConfig = config.pop("validation")

    # Run main hyperparameter optimization experiment.
    run_uri: str = _run(config)
    # Run validation (sensitivity analysis) experiment.
    if validation_config.num_samples > 0:
        validation_config.update({"search_alg": {"_type": "random"}})
        config = OmegaConf.merge(
            config,
            OmegaConf.create(
                {
                    # Take the best hyperparameters from this MLFlow run. And vary random seed to validate
                    # these HPs are stable.
                    "hparams": [run_uri, "params:random_state=tune.randint(0, int(2**32 - 1))"],
                    "tune": {"tune_config": validation_config},
                }
            ),
        )
        _: str = _run(config)


def _run(config: DictConfig) -> str:
    assert config.stage == "search_hp", f"Invalid stage {config.stage}."

    ray.init()
    ray_tune_extensions.add_representers()
    experiment_id: t.Optional[str] = MLflow.create_experiment(name=config.mlflow.experiment_name)

    description: Text = Text.from_chunks(config.mlflow.description, " ".join(sys.argv))
    with mlflow.start_run(description=str(description), experiment_id=experiment_id) as active_run:
        # This MLflow run tracks Ray Tune hyperparameter search. Individual trials won't have their own MLflow runs.
        MLflow.init_run(active_run, set_tags_from_env=True, user_tags=config.mlflow.tags)

        log_deprecate_msg_for_run_inputs(logger)
        IO.save_yaml(
            data=encode(
                {
                    "dataset": config.dataset,
                    "model": config.model,
                    "algorithm": config.tune.tune_config.search_alg.get("_type"),
                    "hparams": config.hparams,
                    "num_trials": config.tune.tune_config.num_samples,
                    "gpu": _gpu_resource(config) > 0,
                }
            ),
            file_path=MLflow.get_artifact_path(active_run) / "run_inputs.yaml",
            raise_on_error=False,
        )
        OmegaConf.save(config, MLflow.get_artifact_path(active_run) / "experiment.yaml", resolve=False)

        artifact_path: Path = MLflow.get_artifact_path(active_run)
        run_id: str = active_run.info.run_id

        ctx = Context(
            Metadata(dataset=config.dataset, model=config.model, run_type=RunType.HPO),
            dataset=build_dataset(config.dataset),
        )
        IO.save_yaml(ctx.dataset.metadata.to_json(), artifact_path / "dataset_info.yaml", raise_on_error=False)

        mlflow.set_tags(
            encode(
                {
                    "dataset": config.dataset,
                    "model": config.model,
                    "run_type": RunType.HPO.value,
                    "algorithm": config.tune.tune_config.search_alg.get("_type"),
                    "task": ctx.dataset.metadata.task.type.value,
                    "framework": "tune",
                }
            )
        )
        mlflow.log_params(
            encode(
                {
                    "dataset": config.dataset,
                    "model": config.model,
                    "num_trials": config.tune.tune_config.num_samples,
                    "algorithm": config.tune.tune_config.search_alg.get("_type"),
                }
            )
        )

        hparams: HParamsSource = config.hparams
        if config.hparams is None:
            hparams = default_hparams(model=config.model, task=ctx.dataset.metadata.task, run_type=RunType.HPO)

        param_space: t.Dict = get_hparams(hparams)
        logger.info("Hyperparameter search space resolved to: '%s'", param_space)

        # Set any `tune_config` parameters before calling  the `_init_search_algorithm` method. The reason for this is
        # there maybe duplicate parameters in the `search_alg` instance that will not be set (e.g.,
        # BasicVariantGenerator's max_concurrent parameter).
        tune_config = _init_tune_config(
            config.tune.tune_config, metric=METRICS.get_primary_metric(ctx.dataset.metadata.task), mode="min"
        )
        run_config = RunConfig(
            name="ray_tune",
            local_dir=artifact_path.as_posix(),
            log_to_file=True,
            callbacks=[RayTuneDriverToMLflowLoggerCallback(tune_config.metric, tune_config.mode)],
        )

        estimator: t.Type[Estimator] = get_estimator(config.model)
        objective_fn = tune.with_parameters(estimator.fit, ctx=ctx)
        if config.tune.trial_resources:
            objective_fn = tune.with_resources(
                objective_fn, OmegaConf.to_container(config.tune.trial_resources, resolve=True)
            )

        tuner = tune.Tuner(objective_fn, param_space=param_space, tune_config=tune_config, run_config=run_config)
        results: ResultGrid = tuner.fit()

        MLflow.set_status_tag_from_trial_counts(len(results), results.num_errors)

        best_trial_metrics: t.Dict = _get_metrics_for_best_trial(results, ctx)
        MLflow.log_metrics(best_trial_metrics)
        _save_best_trial_info(results, artifact_path, best_trial_metrics, active_run)
        _save_summary(artifact_path, active_run)
        print(f"MLFlow run URI: mlflow:///{run_id}")
    ray.shutdown()

    return f"mlflow:///{run_id}"


def _gpu_resource(config: DictConfig) -> float:
    trial_resources: t.Optional[DictConfig] = config.tune.tune_config.get("trial_resources", None)
    if trial_resources is None:
        return 0
    assert isinstance(trial_resources, DictConfig), "Invalid trial resources definition."

    gpu: t.Optional[float] = trial_resources.get("gpu", None)
    if gpu is None or gpu <= 0:
        return 0
    return gpu


def _init_tune_config(cfg: DictConfig, metric: str, mode: str) -> TuneConfig:
    cfg = cfg.copy()
    search_alg: t.Optional[DictConfig] = cfg.pop("search_alg", None)
    #
    tune_config = TuneConfig(
        **OmegaConf.to_container(OmegaConf.merge(cfg, {"metric": metric, "mode": mode}), resolve=True)
    )
    #
    if search_alg is None:
        search_alg = OmegaConf.create({"_type": "random"})
    search_alg_type: str = search_alg.pop("_type", None)

    if search_alg_type in {None, "random", "BasicVariantGenerator"}:
        search_alg.max_concurrent = tune_config.max_concurrent_trials
        tune_config.search_alg = BasicVariantGenerator(**OmegaConf.to_container(search_alg, resolve=True))
    elif search_alg_type == "hyperopt":
        search_alg.update(metric=tune_config.metric, mode=tune_config.mode)
        tune_config.search_alg = HyperOptSearch(**OmegaConf.to_container(search_alg, resolve=True))
    else:
        raise ValueError(f"Unsupported hyperparameter optimization algorithm: {search_alg_type}.")

    return tune_config


def _get_metrics_for_best_trial(results: ResultGrid, ctx: Context) -> t.Dict:
    """Return dictionary that maps a metric name to its value for this task.

    Returns:
        Dictionary that maps a metric name to its value for this task. The metric names are task-specific, e.g.,
            for classification tasks it will include metrics such as `dataset_accuracy`, `train_accuracy` etc.
    """
    best_result: Result = results.get_best_result()
    metrics: t.Dict = copy.deepcopy(best_result.metrics or {})
    missing_metrics: t.Set = {m for m in METRICS[ctx.dataset.metadata.task.type] if m not in metrics}
    if missing_metrics:
        print(f"[WARNING] Missing metrics in the best trial: {missing_metrics}. Program may crash.")
    return metrics


def _save_best_trial_info(results: ResultGrid, local_dir: Path, metrics: t.Dict, active_run: ActiveRun) -> None:
    best_result: Result = results.get_best_result()
    _relative_path: str = Path(best_result.log_dir).relative_to(local_dir).as_posix()
    num_failed_trials: int = results.num_errors
    IO.save_to_file(
        {
            "relative_path": _relative_path,
            "local_path": best_result.log_dir.as_posix(),
            "config": encode(best_result.config),
            "metrics": metrics,
            "num_failed_trials": num_failed_trials,
            "num_successful_trials": len(results) - num_failed_trials,
            "run_uri": f"mlflow:///{active_run.info.run_id}",
            "trial_uri": f"mlflow:///{active_run.info.run_id}/{_relative_path}",
        },
        (local_dir / "best_trial.yaml").as_posix(),
    )


def _save_summary(local_dir: Path, active_run: ActiveRun) -> None:
    IO.save_to_file(Analysis.get_summary(active_run.info.run_id), (local_dir / "summary.yaml").as_posix())


def create_example_config() -> DictConfig:
    """Create a template to be used with `experiment run` command.

    Returns:
        An example configuration file for the `train` experiment.
    """
    # fmt: off
    return OmegaConf.merge(
        _DEFAULT_CONFIGURATION,
        {
            "stage": "search_hp",

            "dataset": "churn_modelling:default",
            "model": "xgboost",
            "hparams": "auto:default:model=xgboost;task=binary_classification;run_type=train",
        },
    )
    # fmt: on


# fmt: off
_DEFAULT_CONFIGURATION = OmegaConf.create(
    {
        "stage": "???",                   # Name of this stage must be `search_hp`.

        "dataset": "???",                 # Dataset fully-qualified name (e.g., {name}:{version}).
        "model": "???",                   # Model name, e.g., `xgboost`.
        "hparams": None,                  # Hyperparameter specs (see HParamsSource docstring for possible values).

        "tune": {                         # Configuration for Ray Tune framework.
            "tune_config": {              # Configuration for TuneConfig instance (__init__ kwargs except `search_alg`).
                "search_alg": {           # Search algorithm specs - everything here is __init__ kwargs except `_type`.
                    "_type": "random"     # Algorithm name (random / hyperopt.)
                },
                "max_concurrent_trials": 0,
                "num_samples": 100
            },
            "trial_resources": {}         # Trial resources to be used with
        },

        "validation": {                   # TuneConfig configuration for validation run (sensitivity analysis).
            "max_concurrent_trials": 0,
            "num_samples": None           # If none or 0, will be disabled.
        },

        "mlflow": {                       # Parameters for MLflow tracking API.
            "experiment_name": "${oc.env:MLFLOW_EXPERIMENT_NAME, null}",
            "description": "",
            "tags": {}
        },
    }
)
# fmt: on
"""Default configuration template (`???` means mandatory value, null - None)."""
