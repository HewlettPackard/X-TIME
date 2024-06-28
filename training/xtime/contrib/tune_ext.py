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
import math
import os
import typing as t
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.entities import Run
from mlflow.utils.file_utils import local_file_uri_to_path
from ray import tune
from ray.tune import Callback
from ray.tune.experiment import Trial
from ray.tune.search import sample
from tqdm import tqdm

from xtime.contrib.mlflow_ext import MLflow
from xtime.datasets import DatasetMetadata
from xtime.io import IO
from xtime.ml import METRICS, Task
from xtime.run import RunType

__all__ = ["gpu_available", "RayTuneDriverToMLflowLoggerCallback", "Analysis", "RandomVarDomain", "add_representers"]


def gpu_available() -> bool:
    """Ray tune specific implementation."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None):
        return True
    return False


def get_trial_dir(
    trial_dir: Path, model_file: str, backup_trial_dir_resolver: t.Optional[t.Callable] = None
) -> t.Optional[Path]:
    """Helper function to identify the directory containing all artifacts for a given Ray Tune trial.

    This is a temporarily ad-hoc solution that we needed due to original location of MLflow artifacts we had in our
    environment. We ran out of space and had to move these artifacts (create a backup copy) some place else. This
    function can be used to identify the actual directory containing trial artifacts (in particular, ML model).

    Args:
        trial_dir: Candidate trial directory. Originally, this is default location resolved ar Mlflow artifact.
        model_file: Model file name that must exist.
        backup_trial_dir_resolver: Optional callback that provides next trial candidate directory.

    Returns:
        Path instance where all mandatory files exist.
    """
    if all([(trial_dir / name).is_file() for name in (model_file, "params.json", "result.json")]):
        return trial_dir
    if backup_trial_dir_resolver:
        return get_trial_dir(backup_trial_dir_resolver(trial_dir), model_file)
    return None


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
        self.metric: str = metric
        self.mode: str = _check_mode(mode)
        self.best_value: float = math.inf if self.mode == "min" else -math.inf
        self.trial_index: int = 0

    def on_trial_result(self, iteration: int, trials: t.List[Trial], trial: Trial, result: t.Dict, **info) -> None:
        self.trial_index += 1

        value = result[self.metric]
        if self.mode == "min":
            self.best_value = min(self.best_value, value)
        else:
            self.best_value = max(self.best_value, value)

        mlflow.log_metric(f"trial_last_{self.metric}", value, self.trial_index)
        mlflow.log_metric(f"trial_best_{self.metric}", self.best_value, self.trial_index)
        mlflow.log_metric("trial_run_time_seconds", result["time_total_s"], self.trial_index)


class Analysis(object):
    @staticmethod
    def get_trial_stats(run: str, **kwargs) -> t.List[t.Dict]:
        """Return descriptive statistics for a given hyperparameter search experiment.

        Not all models can be supported.

        Args:
            run: MLflow run ID that must correspond to Ray Tune experiment.
            **kwargs: Additional (optional arguments):
                - `backup_trial_dir_resolver` Optional callback function to resolve actual directories containing
                    trial data. This is a temporary solution to handle the case when MLflow artifacts are moved to
                    someplace else.
                - `skip_model_stats` If true, do not compute model stats.

        Returns:
            List of dictionaries where each dictionary describes one ray tune trial. it will contain information about
                the model (`model_` prefix), the dataset (`dataset_` prefix), ray tune execution environment (`tune_`
                prefix), MLflow execution environment (`mlflow_` prefix), model input parameters (`param_` prefix) and
                model performance metrics (`metric_` prefix).
        """
        from xtime.estimators.estimator import Model

        mlflow_run_id = run[10:] if run.startswith("mlflow:///") else run
        mlflow_run = mlflow.get_run(mlflow_run_id)
        run_type: RunType = RunType(mlflow_run.data.tags["run_type"])
        if run_type != RunType.HPO:
            raise ValueError(f"Unsupported MLflow run ({mlflow_run.data.tags['run_type']})")
        trials_stats: t.List[t.Dict] = []
        artifact_path: Path = Path(local_file_uri_to_path(mlflow_run.info.artifact_uri))
        task: Task = Task.from_dataset_info(IO.load_yaml(artifact_path / "dataset_info.yaml"))
        trial_dirs: t.List[Path] = [_dir for _dir in (artifact_path / "ray_tune").iterdir() if _dir.is_dir()]

        if kwargs.get("skip_model_stats", False) is True:

            def get_model_stats(*_args, **_kwargs) -> t.Dict:
                return {}
        else:
            if mlflow_run.data.tags["model"] == "xgboost":
                from xtime.contrib.xgboost_ext import get_model_stats
            elif mlflow_run.data.tags["model"] in ("rf", "rf_clf"):
                from xtime.contrib.sklearn_ext import get_model_stats
            else:
                raise ValueError(f"Unsupported model ({mlflow_run.data.tags['model']})")

        for trial_dir in tqdm(
            trial_dirs, total=len(trial_dirs), desc="Hyperparameter search trials", unit="trials", leave=False
        ):
            model_file: str = Model.get_file_name(mlflow_run.data.tags["model"])
            resolved_trial_dir = get_trial_dir(trial_dir, model_file, kwargs.get("backup_trial_dir_resolver", None))
            if resolved_trial_dir is None:
                print(f"WARNING no valid trial directory found (id={mlflow_run_id}, path={trial_dir}).")
                continue
            trial_stats = {
                "model_file": (resolved_trial_dir / model_file).as_posix(),
                "model_name": mlflow_run.data.tags["model"],
                "dataset_name": mlflow_run.data.tags["dataset_name"],
                "dataset_version": mlflow_run.data.tags["dataset_version"],
                "tune_root_path": (artifact_path / "ray_tune").as_posix(),
                "tune_trial_path": resolved_trial_dir.as_posix(),
                "mlflow_run_id": mlflow_run_id,
            }

            try:
                model_stats: t.Dict = get_model_stats(resolved_trial_dir, task.type)
                for k, v in model_stats.items():
                    trial_stats["model_" + k] = v
            except (IOError, EOFError):
                # When there's something wrong with the serialized model.
                print("WARNING can't get model stats, dir=", resolved_trial_dir.as_posix())
                continue

            model_params: t.Dict = IO.load_dict(resolved_trial_dir / "params.json")
            for k, v in model_params.items():
                trial_stats["param_" + k] = v

            metrics: t.Dict = IO.load_dict(resolved_trial_dir / "result.json")
            _known_metrics = {
                "dataset_accuracy",
                "dataset_loss_total",
                "train_accuracy",
                "train_loss_mean",
                "train_loss_total",
                "valid_accuracy",
                "valid_loss_mean",
                "valid_loss_total",
                "test_accuracy",
                "test_loss_mean",
                "test_loss_total",
                "dataset_loss_mean",
            }
            for k, v in metrics.items():
                if k in _known_metrics:
                    trial_stats["metric_" + k] = v

            trials_stats.append(trial_stats)

        return trials_stats

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

            summary["best_run"] = {}
            best_trial: t.Optional[Trial] = experiment.get_best_trial(perf_metric, mode="min")
            if best_trial is not None:
                best_params = IO.load_json((Path(best_trial.logdir) / "params.json").as_posix())
                best_results = IO.load_json((Path(best_trial.logdir) / "result.json").as_posix())
                best_results = {k: best_results[k] for k in perf_metrics if k in best_results}
                summary["best_run"] = {"perf_metric": perf_metric, "parameters": best_params, "results": best_results}

            summary["metric_variations"] = {}
            if experiment.results_df is not None:
                succeeded_trials: pd.DataFrame = experiment.results_df[experiment.results_df[perf_metric].notna()]
                results = succeeded_trials.sort_values([perf_metric], ascending=True)
                for metric in perf_metrics:
                    if metric in results.columns:
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
        from xtime.estimators.estimator import LegacySavedModelInfo

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
        best_trial: t.Optional[Trial] = experiment.get_best_trial(perf_metric, mode="min")
        # Create return object
        model = mlflow_run.data.tags["model"]
        model_file_name: str = LegacySavedModelInfo(model).file_name()
        best_trial_info = {
            "mlflow_run_id": mlflow_run_id,  # MLflow run ID
            "model": model,  # Model name (xgboost, light_gbm_clf, catboost, rf_clf)
            "dataset_info_file": (artifact_path / "dataset_info.yaml").as_posix(),  # Info about dataset
        }
        if best_trial is not None:
            best_trial_info.update(
                {
                    "tune_trial_id": best_trial.trial_id,  # Ray Tune Run ID
                    "trial_path": best_trial.logdir,  # Local path to ray tune trial directory
                }
            )
            if (Path(best_trial.logdir) / "params.json").is_file():
                best_trial_info["params_file"] = "params.json"
            if (Path(best_trial.logdir) / model_file_name).is_file():
                best_trial_info["model_file"] = model_file_name
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

            trials: t.List[Trial] = experiment.trials or []
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


RandomVarDomain = t.TypeVar("RandomVarDomain", bound=sample.Domain)
"""Domain types for random variables (children of `Domain` class) such as `Float`, `Integer` and `Categorical`."""


class YamlEncoder:
    @staticmethod
    def represent(dumper: yaml.representer.BaseRepresenter, rv: RandomVarDomain) -> yaml.nodes.MappingNode:
        """Represent given random variable for yaml dumper."""
        sampler: t.Dict = YamlEncoder.sampler_to_dict(rv.sampler)
        if isinstance(rv, sample.Integer):
            return dumper.represent_mapping(
                "ray.tune.search.sample.Integer", [("lower", rv.lower), ("upper", rv.upper), ("sampler", sampler)]
            )
        if isinstance(rv, sample.Float):
            return dumper.represent_mapping(
                "ray.tune.search.sample.Float", [("lower", rv.lower), ("upper", rv.upper), ("sampler", sampler)]
            )
        if isinstance(rv, sample.Categorical):
            return dumper.represent_mapping(
                "ray.tune.search.sample.Categorical", [("categories", rv.categories), ("sampler", sampler)]
            )
        raise ValueError(f"Unsupported domain ({rv}).")

    @staticmethod
    def construct_float(loader: yaml.loader.Loader, node: yaml.Node) -> t.Any:
        """Reconstruct floating point domain."""
        assert isinstance(node, yaml.MappingNode), "Expecting Mapping node here."
        values: t.Dict = loader.construct_mapping(node, deep=True)
        return YamlEncoder._with_sampler(sample.Float(values["lower"], values["upper"]), values["sampler"])

    @staticmethod
    def construct_integer(loader: yaml.loader.Loader, node: yaml.Node) -> t.Any:
        """Reconstruct integer domain."""
        assert isinstance(node, yaml.MappingNode), "Expecting Mapping node here."
        values: t.Dict = loader.construct_mapping(node, deep=True)
        return YamlEncoder._with_sampler(sample.Integer(values["lower"], values["upper"]), values["sampler"])

    @staticmethod
    def construct_category(loader: yaml.loader.Loader, node: yaml.Node) -> t.Any:
        """Reconstruct categorical domain."""
        assert isinstance(node, yaml.MappingNode), "Expecting Mapping node here."
        values: t.Dict = loader.construct_mapping(node, deep=True)
        return YamlEncoder._with_sampler(sample.Categorical(values["categories"]), values["sampler"])

    @staticmethod
    def _with_sampler(rv: RandomVarDomain, sampler: t.Dict, q: t.Optional[int] = None) -> RandomVarDomain:
        """Add to random variable appropriate sampler based on its dictionary representation.

        Args:
            rv: Random variable.
            sampler: Sampler for this random variable.
            q: If not None, sampler needs to be wrapped with a quantized sampler.

        Returns:
            Input variable (that is modified in-place) with properly set sampler.
        """

        def _quantized(_rv: t.Union[sample.Integer, sample.Float]) -> sample.Domain:
            assert q is None or (
                q is not None and not isinstance(_rv, sample.Categorical)
            ), f"Samplers for categorical variables cannot be quantized (var={_rv})."
            return _rv if q is None else _rv.quantized(q)

        sampler_t = sampler["_sampler"]
        if sampler_t == "none":
            return rv
        if sampler_t == "grid":
            assert isinstance(rv, sample.Categorical), "Only categorical variables can have grid sampler."
            return rv.grid()
        if sampler_t == "uniform":
            return _quantized(rv.uniform())

        assert isinstance(
            rv, (sample.Integer, sample.Float)
        ), f"Only numerical variables can have loguniform and normal samplers (var={rv}, sampler={sampler}, q={q})."
        if sampler_t == "loguniform":
            return _quantized(rv.loguniform(sampler["base"]))
        if sampler_t == "quantized":
            return YamlEncoder._with_sampler(rv, sampler["sampler"], sampler["q"])

        assert isinstance(
            rv, sample.Float
        ), f"Only float variables can have normal samplers (var={rv}, sampler={sampler}, q={q})."
        if sampler_t == "normal":
            return _quantized(rv.normal(sampler["mean"], sampler["sd"]))

        raise ValueError(f"Unexpected sampler ({sampler}).")

    @staticmethod
    def sampler_to_dict(sampler: sample.Sampler) -> t.Dict:
        """Represent a sampler using python dictionary.

        Args:
            sampler: Ray tune sampler.
        Returns:
            Dictionary with keys describing this sampler and that later can be used to uniquely recreate this sampler.
        """
        names = [
            (sample.Grid, "grid"),
            (sample.Quantized, "quantized"),
            (sample.Normal, "normal"),
            (sample.Uniform, "uniform"),
            (sample.LogUniform, "loguniform"),
        ]
        sdict: t.Dict[str, t.Any] = {"_sampler": "none"}
        for stype, sname in names:
            if isinstance(sampler, stype):
                sdict["_sampler"] = sname
                break
        if sampler is not None and sdict["_sampler"] == "none":
            raise ValueError(f"Unsupported sampler: {sampler}.")

        if isinstance(sampler, sample.LogUniform):
            sdict.update(base=sampler.base)
        elif isinstance(sampler, sample.Normal):
            sdict.update(mean=sampler.mean, sd=sampler.sd)
        elif isinstance(sampler, sample.Quantized):
            sdict.update(q=sampler.q, sampler=YamlEncoder.sampler_to_dict(sampler.sampler))

        return sdict


def add_representers() -> None:
    if sample.Integer not in yaml.SafeDumper.yaml_representers:
        for var_type in (sample.Categorical, sample.Integer, sample.Float):
            yaml.add_representer(var_type, YamlEncoder.represent, Dumper=yaml.SafeDumper)

        loader = yaml.SafeLoader
        yaml.add_constructor("ray.tune.search.sample.Categorical", YamlEncoder.construct_category, Loader=loader)
        yaml.add_constructor("ray.tune.search.sample.Integer", YamlEncoder.construct_integer, Loader=loader)
        yaml.add_constructor("ray.tune.search.sample.Float", YamlEncoder.construct_float, Loader=loader)
