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
import typing as t
from pathlib import Path

import catboost as cb

from xtime.contrib.tune_ext import gpu_available
from xtime.datasets import Dataset, DatasetMetadata, DatasetSplit
from xtime.io import IO
from xtime.ml import TaskType

from .estimator import Estimator


class CatboostEstimator(Estimator):
    NAME = "catboost"

    LOSS_FUNCTIONS = {
        TaskType.BINARY_CLASSIFICATION: "CrossEntropy",
        TaskType.MULTI_CLASS_CLASSIFICATION: "MultiClass",
        TaskType.REGRESSION: "RMSE",
    }

    def __init__(self, params: t.Dict, dataset_metadata: DatasetMetadata) -> None:
        super().__init__()
        params = copy.deepcopy(params)
        params.update(
            {
                "train_dir": IO.work_dir() / "catboost_info",
                "loss_function": CatboostEstimator.LOSS_FUNCTIONS[dataset_metadata.task.type],
                "verbose": False,
            }
        )
        if gpu_available():
            params.update({"task_type": "GPU", "devices": "0"})

        self.params = params
        self.model = self.make_model(dataset_metadata, cb.CatBoostClassifier, cb.CatBoostRegressor, params)

    def save_model(self, save_dir: Path) -> None:
        self.model.save_model((save_dir / "model.bin").as_posix())

    def fit_model(self, dataset: Dataset, **kwargs) -> None:
        train_split = dataset.split(DatasetSplit.TRAIN)
        eval_split = dataset.split(DatasetSplit.EVAL_SPLITS)

        kwargs = copy.deepcopy(kwargs)
        kwargs.update(
            {
                "cat_features": dataset.metadata.categorical_feature_names(),
                "early_stopping_rounds": 15,
                "eval_set": [(eval_split.x, eval_split.y)],
                "verbose": True,
            }
        )
        self.model.fit(train_split.x, train_split.y, **kwargs)
