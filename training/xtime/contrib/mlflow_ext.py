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
from xtime.datasets import parse_dataset_name
from xtime.ml import Task
from xtime.run import RunType

__all__ = ["MLflow"]


class MLflow(object):
    @staticmethod
    def set_tags(
        dataset: t.Optional[str] = None, run_type: t.Optional[RunType] = None, task: t.Optional[Task] = None, **kwargs
    ) -> None:
        if dataset:
            dataset_name, dataset_version = parse_dataset_name(dataset)
            mlflow.set_tags({"dataset_name": dataset_name, "dataset_version": dataset_version})
        if run_type:
            mlflow.set_tag("run_type", run_type.value)
        if task:
            mlflow.set_tag("task", task.type.value)
        mlflow.set_tags(kwargs)

    @staticmethod
    def get_experiment_ids(client: t.Optional[MlflowClient] = None) -> t.List[str]:
        if client is None:
            client = MlflowClient()

        ids: t.List[str] = []
        page_token: t.Optional[str] = None
        while True:
            experiments: PagedList[Experiment] = client.list_experiments(page_token=page_token)
            ids.extend(e.experiment_id for e in experiments)
            page_token = experiments.token
            if not page_token:
                break

        return ids

    @staticmethod
    def get_runs(
        client: t.Optional[MlflowClient] = None,
        experiment_ids: t.Optional[t.List[str]] = None,
        filter_string: t.Optional[str] = None,
    ) -> t.List[Run]:
        if client is None:
            client = MlflowClient()

        runs: t.List[Run] = []
        while True:
            _runs: PagedList[Run] = client.search_runs(experiment_ids=experiment_ids, filter_string=filter_string)
            runs.extend(_runs)
            page_token = _runs.token
            if not page_token:
                break

        return runs

    @staticmethod
    def create_experiment(client: t.Optional[MlflowClient] = None) -> None:
        if client is None:
            client = MlflowClient()

        from mlflow.tracking import _EXPERIMENT_NAME_ENV_VAR

        name = os.environ.get(_EXPERIMENT_NAME_ENV_VAR, None)
        if name and client.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name)

    @staticmethod
    def get_tags_from_env() -> t.Dict:
        return hparams.from_string(os.environ.get("MLFLOW_TAGS", None))

    @staticmethod
    def get_artifact_path(run: t.Optional[Run] = None, ensure_exists: bool = True) -> Path:
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
    def init_run(run: t.Optional[Run]):
        _ = MLflow.get_artifact_path(run, ensure_exists=True)
