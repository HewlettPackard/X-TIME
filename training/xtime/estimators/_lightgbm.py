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

import lightgbm as lgb

from xtime.contrib.tune_ext import gpu_available
from xtime.datasets import Dataset, DatasetMetadata, DatasetSplit
from xtime.ml import TaskType

from .estimator import Estimator


class LightGBMClassifierEstimator(Estimator):
    NAME = "lightgbm"

    OBJECTIVES: t.Dict[TaskType, str] = {
        TaskType.BINARY_CLASSIFICATION: "binary",
        TaskType.MULTI_CLASS_CLASSIFICATION: "multiclass",
        TaskType.REGRESSION: "regression",
    }

    def __init__(self, params: t.Dict, dataset_metadata: DatasetMetadata) -> None:
        super().__init__()
        params = copy.deepcopy(params)
        params["objective"] = LightGBMClassifierEstimator.OBJECTIVES[dataset_metadata.task.type]
        if dataset_metadata.task.type == TaskType.MULTI_CLASS_CLASSIFICATION:
            params["num_class"] = dataset_metadata.task.num_classes
        if gpu_available():
            params["device"] = "GPU"

        self.params = params
        self.model: lgb.LGBMModel = self.make_model(dataset_metadata, lgb.LGBMClassifier, lgb.LGBMRegressor, params)

    def save_model(self, save_dir: Path) -> None:
        self.model.booster_.save_model(save_dir / "model.txt")

    def fit_model(self, dataset: Dataset, **kwargs) -> None:
        train_split = dataset.split(DatasetSplit.TRAIN)
        eval_split = dataset.split(DatasetSplit.EVAL_SPLITS)

        kwargs = copy.deepcopy(kwargs)
        kwargs.update(
            {
                "eval_set": [(eval_split.x, eval_split.y)],
                "feature_name": dataset.metadata.feature_names(),
                "categorical_feature": dataset.metadata.categorical_feature_names(),
            }
        )
        self.model.fit(train_split.x, train_split.y, **kwargs)
