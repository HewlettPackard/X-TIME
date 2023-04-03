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

import sys
import typing as t

import mlflow

from xtime.contrib.mlflow_ext import MLflow
from xtime.datasets import build_dataset
from xtime.estimators import Estimator, get_estimator
from xtime.hparams import get_hparams
from xtime.io import IO
from xtime.run import Context, Metadata, RunType


def train(dataset: str, model: str, hparams: t.Union[t.Dict, str, t.Tuple[str], t.List[str]]) -> None:
    """Train a model for a given problem using default, HP-optimized or pre-defined parameters.

    Enable GPUs by setting CUDA_VISIBLE_DEVICES variable (e.g., CUDA_VISIBLE_DEVICES=0 python -m xtime ...).
    """
    MLflow.create_experiment()
    with mlflow.start_run(description=" ".join(sys.argv)) as active_run:
        # This MLflow run tracks model training.
        MLflow.init_run(active_run)
        IO.save_yaml(
            {"dataset": dataset, "model": model, "hparams": hparams},
            MLflow.get_artifact_path(active_run) / "run_inputs.yaml",
        )
        context = Context(
            metadata=Metadata(dataset=dataset, model=model, run_type=RunType.TRAIN), dataset=build_dataset(dataset)
        )
        hparams: t.Dict = get_hparams(hparams, context)
        estimator: t.Type[Estimator] = get_estimator(model)
        _ = estimator.fit(hparams, context)
        print(f"MLflowRun uri=mlflow:///{active_run.info.run_id}")
