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
import typing as t
from pathlib import Path

from xgboost.sklearn import XGBClassifier, XGBModel, XGBRegressor

from xtime.contrib.tune_ext import gpu_available
from xtime.datasets import Dataset, DatasetMetadata, DatasetSplit
from xtime.ml import TaskType

from ..errors import DatasetError
from .estimator import Estimator

logger = logging.getLogger(__name__)


class XGBoostEstimator(Estimator):
    """
    Categorical data:
        https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
        Supported tree methods are `gpu_hist`, `approx`, and `hist` (e.g., tree_method="hist")
    """

    NAME = "xgboost"

    OBJECTIVES: t.Dict[TaskType, str] = {
        # Logistic Regression for binary classification, output probability.
        TaskType.BINARY_CLASSIFICATION: "binary:logistic",
        TaskType.MULTI_CLASS_CLASSIFICATION: "multi:softproba",
        TaskType.REGRESSION: "reg:squarederror",
    }

    EVAL_METRICS: t.Dict[TaskType, t.List[str]] = {
        # `error`: 1.0 - accuracy, `log_Loss`: negative log likelihood
        TaskType.BINARY_CLASSIFICATION: ["error", "logloss"],
        # `merror`: 1.0 - accuracy, `mlogloss`: cross entropy loss
        TaskType.MULTI_CLASS_CLASSIFICATION: ["merror", "mlogloss"],
        TaskType.REGRESSION: ["rmse"],
    }

    def __init__(self, params: t.Dict, dataset_metadata: DatasetMetadata) -> None:
        super().__init__()
        params = copy.deepcopy(params)
        params.update(
            {
                "objective": XGBoostEstimator.OBJECTIVES[dataset_metadata.task.type],
                "enable_categorical": dataset_metadata.has_categorical_features(),
                # The selection of these metrics does not affect objective. We use these to score the progress
                # guided by the objective. In particular, the last metric is used for early stopping.
                "eval_metric": XGBoostEstimator.EVAL_METRICS[dataset_metadata.task.type],
            }
        )
        if gpu_available():
            logger.info("GPU enabled for XGBoost Estimator: device=cuda, tree_method=hist.")
            params.update({"device": "cuda", "tree_method": "hist"})

        self.params = params
        self.model: XGBModel = self.make_model(dataset_metadata, XGBClassifier, XGBRegressor, params)

    def save_model(self, save_dir: Path) -> None:
        self.model.save_model(save_dir / "model.ubj")

    def fit_model(self, dataset: Dataset, **kwargs) -> None:
        # Early stopping is active only when at least one dataset split is present in `eval_set`. Early stopping
        # will be determined using last dataset split in `eval_set` and last metric in `eval_metric`. Generally, it
        # is recommended to use 10% of `n_estimators` parameter.
        # Actual number of trees can differ from `clf.n_estimators`. Total number of trees will be
        # clf.best_ntree_limit + early_stopping_rounds (predict will use best_ntree_limit to use the best model).
        # The `clf.best_iteration` will point to the best iteration (clf.best_ntree_limit - 1).
        kwargs = copy.deepcopy(kwargs)
        if kwargs.get("early_stopping_rounds", None) is None:
            kwargs["early_stopping_rounds"] = 15
            if "n_estimators" in self.params:
                kwargs["early_stopping_rounds"] = max(15, int(0.1 * self.params["n_estimators"]))
        # This parameter is deprecated in `fit` method, so we set it via `set_params`.
        self.model.set_params(early_stopping_rounds=kwargs.pop("early_stopping_rounds"))

        train_split = dataset.split(DatasetSplit.TRAIN)
        if train_split is None:
            raise DatasetError.missing_train_split(dataset.metadata.name)
        eval_set = [(train_split.x, train_split.y)]  # validation_0

        eval_split = dataset.split(DatasetSplit.EVAL_SPLITS)
        if eval_split is not None:
            eval_set.append((eval_split.x, eval_split.y))  # validation_1
        kwargs.update(
            eval_set=eval_set,
            verbose=kwargs.get("verbose", False),
        )
        self.model.fit(train_split.x, train_split.y, **kwargs)
