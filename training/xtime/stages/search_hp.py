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
import os
import sys
import typing as t
from pathlib import Path

import mlflow
import ray
from mlflow import ActiveRun
from ray import tune
from ray.air import Result, RunConfig
from ray.tune import ResultGrid, TuneConfig
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch

import xtime.contrib.tune_ext as ray_tune_extensions
import xtime.hparams as hp
from xtime.contrib.mlflow_ext import MLflow
from xtime.contrib.tune_ext import Analysis, RayTuneDriverToMLflowLoggerCallback
from xtime.datasets import Dataset
from xtime.estimators import Estimator, get_estimator
from xtime.hparams import HParamsSource, get_hparams
from xtime.hparams import from_string as hparams_from_string
from xtime.io import IO, encode
from xtime.ml import METRICS
from xtime.run import Context, Metadata, RunType

logger = logging.getLogger(__name__)


def search_hp(
    dataset: str,
    model: str,
    algorithm: str,
    hparams: t.Optional[HParamsSource],
    num_trials: int,
    gpu: float = 0,
    fit_params: str = "",
) -> str:
    estimator: t.Type[Estimator] = get_estimator(model)

    ray.init(include_dashboard=True, dashboard_host="0.0.0.0")
    ray_tune_extensions.add_representers()
    MLflow.create_experiment()
    with mlflow.start_run(description=" ".join(sys.argv)) as active_run:
        # This MLflow run tracks Ray Tune hyperparameter search. Individual trials won't have their own MLflow runs.
        MLflow.init_run(active_run)
        IO.save_yaml(
            data={
                "dataset": dataset,
                "model": model,
                "algorithm": algorithm,
                "hparams": hparams,
                "num_trials": num_trials,
                "gpu": gpu,
            },
            file_path=MLflow.get_artifact_path(active_run) / "run_inputs.yaml",
            raise_on_error=False,
        )
        artifact_path: Path = MLflow.get_artifact_path(active_run)
        run_id: str = active_run.info.run_id

        ctx = Context(Metadata(dataset=dataset, model=model, run_type=RunType.HPO), dataset=Dataset.create(dataset))

        fit_params = fit_params or ""
        if fit_params:
            ctx.metadata.fit_params.update(hparams_from_string(fit_params))
            logger.info("Update fit params (%s).", ctx.metadata.fit_params)

        IO.save_yaml(ctx.dataset.metadata.to_json(), artifact_path / "dataset_info.yaml", raise_on_error=False)
        _set_tags(
            dataset=dataset,
            model=model,
            run_type=RunType.HPO,
            algorithm=algorithm,
            task=ctx.dataset.metadata.task,
            framework="tune",
        )
        mlflow.log_params({"dataset": dataset, "model": model, "algorithm": algorithm, "num_trials": num_trials})

        if hparams is None:
            hparams = f"auto:default:model={model};task={ctx.dataset.metadata.task.type.value};run_type=hpo"
            logger.info("Hyperparameters are not provided, using default ones: '%s'.", hparams)
        param_space: t.Dict = get_hparams(hparams)
        logger.info("Hyperparameter search space resolved to: '%s'", param_space)

        # Set any `tune_config` parameters before calling  the `_init_search_algorithm` method. The reason for this is
        # there maybe duplicate parameters in the `search_alg` instance that will not be set (e.g.,
        # BasicVariantGenerator's max_concurrent parameter).
        tune_config = _init_search_algorithm(
            # The `max_concurrent_trials` can be overriden in `_init_search_algorithm`
            TuneConfig(
                metric=METRICS.get_primary_metric(ctx.dataset.metadata.task),
                mode="min",
                num_samples=num_trials,
                max_concurrent_trials=0,
            ),
            algorithm,
        )
        assert tune_config.metric is not None and tune_config.mode is not None, "Internal error (metric not specified)."
        run_config = RunConfig(
            name="ray_tune",
            local_dir=artifact_path.as_posix(),
            log_to_file=True,
            callbacks=[RayTuneDriverToMLflowLoggerCallback(tune_config.metric, tune_config.mode)],
        )

        objective_fn = tune.with_parameters(estimator.fit, ctx=ctx)
        if gpu > 0:
            objective_fn = tune.with_resources(objective_fn, {"gpu": gpu})
        tuner = tune.Tuner(objective_fn, param_space=param_space, tune_config=tune_config, run_config=run_config)
        results: ResultGrid = tuner.fit()

        _set_run_status(results)
        best_trial_metrics: t.Dict = _get_metrics_for_best_trial(results, ctx)
        MLflow.log_metrics(best_trial_metrics)
        _save_best_trial_info(results, artifact_path, best_trial_metrics, active_run)
        _save_summary(artifact_path, active_run)
        print(f"MLFlow run URI: mlflow:///{active_run.info.run_id}")
    ray.shutdown()

    return f"mlflow:///{run_id}"


def _init_search_algorithm(tune_config: TuneConfig, algorithm: str) -> TuneConfig:
    """
    - Setting `tune_config.max_concurrent_trials` wraps search algorithm with `ConcurrencyLimiter`.
    """
    # None and 0 means the same thing - no limit, so set to 0 to simplify subsequent comparison.
    if tune_config.max_concurrent_trials is None:
        tune_config.max_concurrent_trials = 0
    max_concurrent_trials: int = tune_config.max_concurrent_trials

    if algorithm == "random":
        tune_config.search_alg = BasicVariantGenerator(random_state=1, max_concurrent=max_concurrent_trials)
    elif algorithm == "hyperopt":
        tune_config.search_alg = HyperOptSearch(
            metric=tune_config.metric, mode=tune_config.mode, n_initial_points=20, random_state_seed=1
        )
        if max_concurrent_trials == 0:
            logger.info("TuneConfig's max_concurrent_trials is unlimited - setting to 2 for HyperOptSearch algorithm.")
            tune_config.max_concurrent_trials = 2
    else:
        raise ValueError(f"Unsupported hyperparameter optimization algorithm: {algorithm}")
    return tune_config


def _set_tags(**tags) -> None:
    if "MLFLOW_TAGS" in os.environ:
        tags = copy.deepcopy(tags)
        tags.update(hp.get_hparams(f"params:{os.environ['MLFLOW_TAGS']}"))
    MLflow.set_tags(**tags)


def _set_run_status(results: ResultGrid) -> None:
    num_failed_trials: int = results.num_errors
    if num_failed_trials == 0:
        mlflow.set_tag("status", "COMPLETED")
    elif num_failed_trials == len(results):
        mlflow.set_tag("status", "FAILED")
    else:
        mlflow.set_tag("status", "PARTIALLY_COMPLETED")


def _get_metrics_for_best_trial(results: ResultGrid, ctx: Context) -> t.Dict:
    """Return dictionary that maps a metric name to its value for this task.

    Returns:
        Dictionary that maps a metric name to its value for this task. The metric names are task-specific, e.g.,
            for classification tasks it will include metrics such as `dataset_accuracy`, `train_accuracy` etc.
    """
    best_result: Result = results.get_best_result()
    metrics: t.Dict = copy.deepcopy(best_result.metrics or {})

    if ctx.dataset is not None:
        expected_metrics: t.List[str] = METRICS[ctx.dataset.metadata.task.type]
        missing_metrics: t.Set = {m for m in expected_metrics if m not in metrics}
        if missing_metrics:
            logger.warning(
                "Expected metrics not found in best trial (log_dir=%s): dataset=%s, task=%s, expected_metrics=%s, "
                "missing_metrics=%s.",
                best_result.log_dir,
                ctx.dataset.metadata.name,
                ctx.dataset.metadata.task.type,
                expected_metrics,
                missing_metrics,
            )

    return metrics


def _save_best_trial_info(results: ResultGrid, local_dir: Path, metrics: t.Dict, active_run: ActiveRun) -> None:
    best_result: Result = results.get_best_result()

    _relative_path: str = ""
    local_path: str = ""
    trial_uri: str = ""

    if best_result.log_dir is not None:
        log_dir: Path = best_result.log_dir
        _relative_path = Path(log_dir).relative_to(local_dir).as_posix()
        local_path = log_dir.as_posix()
        trial_uri = f"mlflow:///{active_run.info.run_id}/{_relative_path}"

    num_failed_trials: int = results.num_errors
    IO.save_to_file(
        {
            "relative_path": _relative_path,
            "local_path": local_path,
            "config": encode(best_result.config),
            "metrics": encode(metrics),
            "num_failed_trials": num_failed_trials,
            "num_successful_trials": len(results) - num_failed_trials,
            "run_uri": f"mlflow:///{active_run.info.run_id}",
            "trial_uri": trial_uri,
        },
        (local_dir / "best_trial.yaml").as_posix(),
    )


def _save_summary(local_dir: Path, active_run: ActiveRun) -> None:
    IO.save_to_file(Analysis.get_summary(active_run.info.run_id), (local_dir / "summary.yaml").as_posix())
