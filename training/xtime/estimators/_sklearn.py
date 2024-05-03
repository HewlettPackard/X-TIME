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
import pickle
import typing as t
from pathlib import Path

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from xtime.datasets import Dataset, DatasetMetadata, DatasetSplit

from .estimator import Estimator

__all__ = ["DummyEstimator", "RandomForestEstimator"]

from ..errors import DatasetError


class ScikitLearnEstimator(Estimator):
    def __init__(self, params: t.Dict, dataset_metadata: DatasetMetadata, classifier_cls, regressor_cls) -> None:
        super().__init__()
        params = copy.deepcopy(params)

        self.params = params
        self.model = self.make_model(dataset_metadata, classifier_cls, regressor_cls, params)

    def save_model(self, save_dir: Path) -> None:
        with open(save_dir / "model.pkl", "wb") as file:
            pickle.dump(self.model, file)

    def fit_model(self, dataset: Dataset, **kwargs) -> None:
        train_split = dataset.split(DatasetSplit.TRAIN)
        if train_split is None:
            raise DatasetError.missing_train_split(dataset.metadata.name)
        self.model.fit(train_split.x, train_split.y, **kwargs)


class DummyEstimator(ScikitLearnEstimator):
    NAME = "dummy"

    def __init__(self, params: t.Dict, dataset_metadata: DatasetMetadata) -> None:
        super().__init__(params, dataset_metadata, DummyClassifier, DummyRegressor)


class RandomForestEstimator(ScikitLearnEstimator):
    NAME = "rf"

    def __init__(self, params: t.Dict, dataset_metadata: DatasetMetadata) -> None:
        super().__init__(params, dataset_metadata, RandomForestClassifier, RandomForestRegressor)
