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
import pickle
import typing as t
from pathlib import Path

import pandas as pd

try:
    from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
    from cuml.ensemble.randomforest_common import BaseRandomForestModel
except ImportError:
    from xtime.errors import EstimatorError

    raise EstimatorError.library_not_installed("RandomForestEstimator", "cuml", ["rapids-12"])

from xtime.datasets import Dataset, DatasetMetadata, DatasetSplit
from xtime.errors import DatasetError
from xtime.estimators import Estimator

__all__ = ["RandomForestEstimator"]

logger = logging.getLogger(__name__)


class RandomForestEstimator(Estimator):
    NAME = "rapids-rf"

    def __init__(self, params: t.Dict, dataset_metadata: DatasetMetadata) -> None:
        super().__init__()
        params = copy.deepcopy(params)
        if "n_streams" not in params:
            logger.warning(
                "The 'n_streams' parameter (number of parallel streams used for forest building) is not specified. "
                "Default value (which is most likely 4) may not be the optimal, and values in 4-10 range may speed-up "
                "the forest building process."
            )

        self.params = params
        self.model: BaseRandomForestModel = self.make_model(
            dataset_metadata, RandomForestClassifier, RandomForestRegressor, params
        )

    def save_model(self, save_dir: Path) -> None:
        with open(save_dir / "model.pkl", "wb") as file:
            pickle.dump(self.model, file)

    def fit_model(self, dataset: Dataset, **kwargs) -> None:
        train_split = dataset.splits[DatasetSplit.TRAIN]
        if train_split is None:
            raise DatasetError.missing_train_split(dataset.metadata.name)

        if dataset.metadata.task.type.classification():
            if not isinstance(train_split.y, pd.Series):
                logger.warning(
                    "Cannot validate data types of labels (expected pandas series, but type is '%s').",
                    type(train_split.y),
                )
            elif not train_split.y.dtype == "int32":
                logger.warning("Data types of labels is '%s' (expecting 'int32').", train_split.y.dtype)

        for feature, dtype in zip(train_split.x.columns, train_split.x.dtypes):
            if dtype not in ["float32"]:
                logger.warning("Feature ('%s') data type ('%s') is not 'float32'.", feature, dtype)

        self.model.fit(train_split.x, train_split.y, **kwargs)
