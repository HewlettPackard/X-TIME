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

from lightgbm.sklearn import LGBMClassifier, LGBMModel, LGBMRegressor

from xtime.contrib.tune_ext import gpu_available
from xtime.datasets import Dataset, DatasetMetadata, DatasetSplit
from xtime.ml import ClassificationTask, TaskType

from ..errors import DatasetError
from .estimator import Estimator


class LightGBMClassifierEstimator(Estimator):
    NAME = "lightgbm"

    OBJECTIVES: t.Dict[TaskType, str] = {
        TaskType.BINARY_CLASSIFICATION: "binary",
        TaskType.MULTI_CLASS_CLASSIFICATION: "multiclass",
        TaskType.REGRESSION: "regression",
    }

    def __init__(self, params: t.Dict, dataset_metadata: DatasetMetadata) -> None:
        """
        https://lightgbm.readthedocs.io/en/latest/Parameters.html
        """
        super().__init__()
        params = copy.deepcopy(params)
        params["objective"] = LightGBMClassifierEstimator.OBJECTIVES[dataset_metadata.task.type]
        if dataset_metadata.task.type == TaskType.MULTI_CLASS_CLASSIFICATION:
            task: ClassificationTask = t.cast(ClassificationTask, dataset_metadata.task)
            params["num_class"] = task.num_classes
        if gpu_available():
            params["device"] = "GPU"

        self.params = params
        self.model: LGBMModel = self.make_model(dataset_metadata, LGBMClassifier, LGBMRegressor, params)

    def save_model(self, save_dir: Path) -> None:
        self.model.booster_.save_model(save_dir / "model.txt")

    def fit_model(self, dataset: Dataset, **kwargs) -> None:
        kwargs = copy.deepcopy(kwargs)

        train_split = dataset.split(DatasetSplit.TRAIN)
        if train_split is None:
            raise DatasetError.missing_train_split(dataset.metadata.name)

        kwargs.update(
            {
                "feature_name": dataset.metadata.feature_names(),
                "categorical_feature": dataset.metadata.categorical_feature_names(),
            }
        )

        eval_split = dataset.split(DatasetSplit.EVAL_SPLITS)
        if eval_split is not None:
            kwargs["eval_set"] = [(eval_split.x, eval_split.y)]

        self.model.fit(train_split.x, train_split.y, **kwargs)
