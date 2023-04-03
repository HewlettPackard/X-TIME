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
import os
import typing as t
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.entities import Run
from mlflow.utils.file_utils import local_file_uri_to_path
from ray import tune
from ray.tune import Callback
from ray.tune.experiment import Trial

from xtime.contrib.mlflow_ext import MLflow
from xtime.datasets import DatasetMetadata
from xtime.io import IO
from xtime.ml import METRICS
from xtime.run import RunType

__all__ = ["gpu_available", "RayTuneDriverToMLflowLoggerCallback", "Analysis"]


def gpu_available() -> bool:
    """Ray tune specific implementation."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None):
        return True
    return False


class RayTuneDriverToMLflowLoggerCallback(Callback):
    """A callback that Ray Tune can use to report progress to MLflow.

    Usage example:
    >>> from ray.air import RunConfig
    ... run_config = RunConfig(
    ...     local_dir='ray_results',
    ...     callbacks=[RayTuneDriverToMLflowLoggerCallback('eval_loss', 'min')]
    ... )

    Args:
        metric: Metric that is being optimized.
        mode: Either this metric should be minimized ('min') or maximized ('max').
    """

    def __init__(self, metric: str, mode: str) -> None:
        super().__init__()
        self.metric = metric
        self.mode = _check_mode(mode)
        self.best_value: t.Optional[float] = None
        self.trial_index = 0

    def on_trial_result(self, iteration: int, trials: t.List[Trial], trial: Trial, result: t.Dict, **info) -> None:
        self.trial_index += 1

        value = result[self.metric]
        if self.best_value is None:
            self.best_value = value
        elif self.mode == "min":
            self.best_value = min(self.best_value, value)
        else:
            self.best_value = max(self.best_value, value)

        mlflow.log_metric(f"trial_last_{self.metric}", value, self.trial_index)
        mlflow.log_metric(f"trial_best_{self.metric}", self.best_value, self.trial_index)
        mlflow.log_metric("trial_run_time_seconds", result["time_total_s"], self.trial_index)


class Analysis(object):
    @staticmethod
    def get_summary(run: str) -> t.Dict:
        """Get the summary of a run.

        Args:
            run: The MLflow run ID or MLflow URI.
        """
        summary: t.Dict = {}
        mlflow_run_id = run[10:] if run.startswith("mlflow:///") else run
        mlflow_run = mlflow.get_run(mlflow_run_id)
        run_type = RunType(mlflow_run.data.tags["run_type"])
        if run_type == RunType.TRAIN:
            raise NotImplementedError("Implement me (run_type = TRAIN)")
        elif run_type == RunType.HPO:
            artifact_path = MLflow.get_artifact_path(mlflow_run, ensure_exists=False)
            experiment = tune.ExperimentAnalysis((artifact_path / "ray_tune").as_posix())

            ds_metadata = DatasetMetadata.from_json(IO.load_yaml(artifact_path / "dataset_info.yaml"))
            perf_metrics = METRICS[ds_metadata.task.type]
            perf_metric = METRICS.get_primary_metric(ds_metadata.task)
            summary["tags"] = copy.deepcopy(mlflow_run.data.tags)
            summary["problem"] = {"dataset": ds_metadata.task.to_json(), "perf_metric": perf_metric}

            failed_trials: pd.DataFrame = experiment.results_df[experiment.results_df[perf_metric].isna()]
            if len(failed_trials) > 0:
                summary["failed_trials"] = {
                    "num_failed_trials": len(failed_trials),
                    "num_total_trials": len(experiment.results_df),
                    "failed_trials_names": list(failed_trials.index),
                }

            best_trial: Trial = experiment.get_best_trial(perf_metric, mode="min")
            best_params = IO.load_json((Path(best_trial.logdir) / "params.json").as_posix())
            best_results = IO.load_json((Path(best_trial.logdir) / "result.json").as_posix())
            best_results = {k: best_results[k] for k in perf_metrics}
            summary["best_run"] = {"perf_metric": perf_metric, "parameters": best_params, "results": best_results}

            summary["metric_variations"] = {}
            succeeded_trials: pd.DataFrame = experiment.results_df[experiment.results_df[perf_metric].notna()]
            results = succeeded_trials.sort_values([perf_metric], ascending=True)
            for metric in perf_metrics:
                summary["metric_variations"][metric] = {
                    "mean": results[metric].mean().item(),
                    "std": results[metric].std().item(),
                }

            summary["mlflow_run"] = mlflow_run_id
            return summary
        else:
            raise ValueError(f"Unsupported run type ({mlflow_run.data.tags['run_type']})")

    @staticmethod
    def get_best_trial(mlflow_uri: str) -> t.Dict:
        """Return information about the best Ray Tune trial in one MLflow run.

        Prerequisites:
            pip install mlflow ray[tune]
            export MLFLOW_TRACKING_URI=http://10.93.226.108:10000
            export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
        Args:
            mlflow_uri: MLflow run URI in the form `mlflow:///MLFLOW_RUN_ID` or just MLFLOW_RUN_ID
        Returns:
            Dictionary containing the following fields:
                - model_path: Local path to an ML model.
                - params_path: Local path to
                - model: Model name (catboost, xgboost)
        """
        # Extract MLflow run ID from `uri`
        mlflow_run_id = mlflow_uri[10:] if mlflow_uri.startswith("mlflow:///") else mlflow_uri
        # Get MLflow run using run ID
        mlflow_run = mlflow.get_run(mlflow_run_id)
        # This run must be associated with hyperparameter search experiment
        run_type: RunType = RunType(mlflow_run.data.tags["run_type"])
        if run_type != RunType.HPO:
            raise ValueError(f"Unsupported MLflow run ({mlflow_run.data.tags['run_type']})")
        # Path where all files associated with this run are stored
        artifact_path: Path = Path(local_file_uri_to_path(mlflow_run.info.artifact_uri))
        # Ray Tune stores all its files in `ray_tune` subdirectory. Parse and get ray tune run summary.
        experiment = tune.ExperimentAnalysis((artifact_path / "ray_tune").as_posix())
        # We need to retrieve the task for this run in order to identify the target metric.
        ds_metadata = DatasetMetadata.from_json(IO.load_yaml(artifact_path / "dataset_info.yaml"))
        # Get primary ML metric that this task is to be evaluated on.
        perf_metric = METRICS.get_primary_metric(ds_metadata.task)
        # Get the best Ray Tune trial that minimizes given metric
        best_trial: Trial = experiment.get_best_trial(perf_metric, mode="min")
        # Create return object
        model = mlflow_run.data.tags["model"]
        models = {
            "xgboost": "model.ubj",
            "light_gbm_clf": "model.txt",
            "catboost": "model.bin",
            "rf_clf": "model.pkl",
            "rf": "model.pkl",
        }
        best_trial_info = {
            "mlflow_run_id": mlflow_run_id,  # MLflow run ID
            "tune_trial_id": best_trial.trial_id,  # Ray Tune Run ID
            "trial_path": best_trial.logdir,  # Local path to ray tune trial directory
            "model": model,  # Model name (xgboost, light_gbm_clf, catboost, rf_clf)
            "dataset_info_file": (artifact_path / "dataset_info.yaml").as_posix(),  # Info about dataset
        }
        if (Path(best_trial.logdir) / "params.json").is_file():
            best_trial_info["params_file"] = "params.json"
        if (Path(best_trial.logdir) / models[model]).is_file():
            best_trial_info["model_file"] = models[model]
        return best_trial_info

    @staticmethod
    def get_final_trials() -> pd.DataFrame:
        """
        Filter all runs by:
            - `params.config` must not be None
            - `params.algorithm` must be `random`.
        Pre-requisites:
            - Iterate over each MLflow run and see if there are any duplicates on data/model key. If so, raise an
              exception.
        Current constraints:
            - Record key is dataset + model +
        """
        runs: t.List[Run] = MLflow.get_runs(
            experiment_ids=MLflow.get_experiment_ids(),
            filter_string="params.algorithm = 'random' AND params.config LIKE 'mlflow%'",
        )

        # Check if we have multiple runs for dataset/model pair
        cache: t.Dict[str, Run] = {}
        for run in runs:
            key = run.data.params["problem"] + "/" + run.data.params["model"]
            if key in cache:
                _prev_run = cache[key].info
                raise RuntimeError(
                    f"Found multiple runs for problem={run.data.params['problem']}, model={run.data.params['model']}. "
                    f"Previously found MLflow run was run_id={_prev_run.run_id} in {_prev_run.experiment_id} "
                    f"experiment. This MLflow run is run_id={run.info.run_id} in {run.info.experiment_id} experiment."
                )
            cache[key] = run

        # Build a data frame containing results from all Ray Tune trials for the found runs. If no trials failed, then
        # data frame will contain len(runs) * 20 records, where 20 was the default number of trials for best models.
        results: t.List[t.Dict] = []
        for run in runs:
            artifact_path: Path = Path(local_file_uri_to_path(run.info.artifact_uri))
            experiment = tune.ExperimentAnalysis((artifact_path / "ray_tune").as_posix())

            ds_metadata = DatasetMetadata.from_json(IO.load_yaml(artifact_path / "dataset_info.yaml"))
            metrics = ["test_mse"] if ds_metadata.task.type.regression() else ["test_accuracy", "test_loss_mean"]

            trials: t.List[Trial] = experiment.trials
            for trial in trials:
                if trial.status != Trial.TERMINATED:
                    continue
                result = {
                    "dataset": run.data.params["problem"],
                    "model": run.data.params["model"],
                    "trial_dir": trial.logdir,
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_experiment_id": run.info.experiment_id,
                    "trial_id": trial.trial_id,
                }
                result.update({k: trial.last_result[k] for k in metrics})

                models = list(Path(trial.logdir).glob("model[.]*"))
                if len(models) != 1:
                    raise RuntimeError(
                        f"No models found in {trial.logdir} (experiment_id={run.info.experiment_id}, "
                        f"run_id={run.info.run_id}, trial_id={trial.trial_id})"
                    )
                result["model"] = models[0].name

                results.append(result)

        return pd.DataFrame(results)


def _check_mode(mode: str) -> str:
    if mode not in ("min", "max"):
        raise ValueError(f"Invalid mode ({mode}). Expecting one of (`min`, `max`).")
    return mode
