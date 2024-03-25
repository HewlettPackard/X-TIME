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

import os
import typing as t
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Experiment, Run
from mlflow.store.entities import PagedList
from mlflow.utils.file_utils import local_file_uri_to_path

from xtime import hparams
from xtime.datasets import Dataset
from xtime.ml import Task
from xtime.run import RunType

__all__ = ["MLflow"]


class MLflow(object):
    @staticmethod
    def set_tags(
        dataset: t.Optional[str] = None, run_type: t.Optional[RunType] = None, task: t.Optional[Task] = None, **kwargs
    ) -> None:
        """Helper method that sets tags for active MLflow run.

        Args:
            dataset: Dataset name and version in format "name[:version]".
            run_type: Type of this MLflow run (`train`, `hpo` etc.).
            task: Task being solved in this run.
            kwargs: dictionary of additional tags to set.
        """
        if dataset:
            dataset_name, dataset_version = Dataset.parse_name(dataset)
            mlflow.set_tags({"dataset_name": dataset_name, "dataset_version": dataset_version})
        if run_type:
            mlflow.set_tag("run_type", run_type.value)
        if task:
            mlflow.set_tag("task", task.type.value)
        mlflow.set_tags(kwargs)

    @staticmethod
    def get_experiment_ids(client: t.Optional[MlflowClient] = None) -> t.List[str]:
        """Return all MLflow experiment IDs.

        Args:
           client: MLflow client to use. If not provided, a default client will be used.

        Returns:
             List of IDs of experiments recorded in MLflow server. Empty list means there's no experiments.
        """
        if client is None:
            client = MlflowClient()

        ids: t.List[str] = []
        page_token: t.Optional[str] = None
        while True:
            experiments: PagedList[Experiment] = client.search_experiments(page_token=page_token)
            ids.extend(e.experiment_id for e in experiments)
            page_token = experiments.token
            if not page_token:
                break

        return ids

    @staticmethod
    def get_runs(
        client: t.Optional[MlflowClient] = None,
        experiment_ids: t.Optional[t.List[str]] = None,
        filter_string: str = "",
    ) -> t.List[Run]:
        """Return MLflow runs that match given query (filter string).

        Args:
            client: MLflow client to use. If not provided, a default client will be used.
            experiment_ids: List of IDs of experiments within which runs will be searched.
            filter_string: Search query in format accepted by MLflow API. Consult MLflow API documentation on exact
                format of this string.

        Returns:
            List of found MLflow runs.
        """
        if client is None:
            client = MlflowClient()

        runs: t.List[Run] = []
        page_token: t.Optional[str] = None
        while True:
            _runs: PagedList[Run] = client.search_runs(
                experiment_ids=experiment_ids if experiment_ids is not None else MLflow.get_experiment_ids(client),
                filter_string=filter_string,
                page_token=page_token,
            )
            runs.extend(_runs)
            page_token = _runs.token
            if not page_token:
                break

        return runs

    @staticmethod
    def create_experiment(client: t.Optional[MlflowClient] = None) -> None:
        """Create a new MLflow experiment with name specified in `MLFLOW_EXPERIMENT_NAME` environment variable.

        Args:
            client: MLflow client to use. If not provided, a default client will be used.
        """
        if client is None:
            client = MlflowClient()

        from mlflow.tracking import _EXPERIMENT_NAME_ENV_VAR

        name = os.environ.get(_EXPERIMENT_NAME_ENV_VAR, None)
        if name and client.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name)

    @staticmethod
    def get_tags_from_env() -> t.Dict:
        """Return dictionary of tags specified via environment.

        Users can provide addition tags to be associated with MLflow runs. It's the same format that hyperparameters are
            used when defined in strings: `export MLFLOW_TAGS="params:tag1=value1;tag2=value2"`.

        Returns:
            Dictionary or tags or empty dictionary.
        """
        return hparams.from_string(os.environ.get("MLFLOW_TAGS", None))

    @staticmethod
    def get_artifact_path(run: t.Optional[Run] = None, ensure_exists: bool = True) -> Path:
        """Return path to artifact directory for given MLflow run.

        Args:
            run: MLflow run or none. If none, currently active MLflow run will be used.
            ensure_exists: If true and path does not exist, it will be created.

        Returns:
            Path to artifact directory (that may or may not exist depending on this run and method parameters).
        """
        if run is not None:
            artifact_uri = run.info.artifact_uri
        elif mlflow.active_run() is not None:
            artifact_uri = mlflow.get_artifact_uri()
        else:
            raise RuntimeError("Can't get MLflow artifact path: no run provided and not active run found.")
        local_dir = Path(local_file_uri_to_path(artifact_uri))
        if ensure_exists:
            local_dir.mkdir(parents=True, exist_ok=True)
        return local_dir

    @staticmethod
    def init_run(run: t.Optional[Run]) -> None:
        """Initialize MLflow run.

        For now, it does not do a lot of things, mainly ensuring that the artifact directory exists. So, it's a wrapper
        over MLflow.get_artifact_path method to better communicate the usage scenarios of this method.

        Args:
            run: MLflow run to initialize. If none, currently active run will be used.
        """
        _ = MLflow.get_artifact_path(run, ensure_exists=True)

    @staticmethod
    def log_metrics(metrics: t.Dict[str, t.Any]) -> None:
        """Log metrics with current MLflow run ignoring some metrics and checking for MLflow exceptions.

        This function logs metrics with current MLflow run. These metrics can come from different platforms, such as
        Ray Tune. Certain metrics are ignored. Only metrics with certain value types are logged. MLflow exceptions
        are caught on a per-metric basis and are ignored.

        Args:
            metrics: Dictionary with metrics from frameworks such as Ray Tune.
        """
        # Some metrics produced by Ray Tune we are not interested in.
        _metrics_to_ignore = {
            "timesteps_total",
            "time_this_iter_s",
            "timesteps_total",
            "episodes_total",
            "training_iteration",
            "timestamp",
            "time_total_s",
            "pid",
            "time_since_restore",
            "timesteps_since_restore",
            "iterations_since_restore",
            "warmup_time",
        }
        for name, value in metrics.items():
            try:
                if isinstance(value, (int, float)) and name not in _metrics_to_ignore:
                    mlflow.log_metric(name, value)
            except mlflow.MlflowException:
                continue
