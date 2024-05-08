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
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import Booster
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from xtime.datasets import Dataset, DatasetMetadata, DatasetSplit
from xtime.estimators._catboost import CatboostEstimator
from xtime.estimators._lightgbm import LightGBMClassifierEstimator
from xtime.estimators._sklearn import DummyEstimator, RandomForestEstimator
from xtime.estimators._xgboost import XGBoostEstimator
from xtime.estimators.estimator import Estimator, LegacySavedModelInfo, Model
from xtime.ml import ClassificationTask, Feature, FeatureType, RegressionTask, TaskType
from xtime.run import Context, Metadata, RunType


class TestModelLoader(TestCase):
    @staticmethod
    def make_classification_dataset(num_samples: int = 500, num_features: int = 20, num_classes: int = 3) -> Dataset:
        """Create random classification datasets.

        Args:
            num_samples: Dataset size (total across train/test split).
            num_features: Number of features.
            num_classes: Number of classes, must be >= 2.

        Returns:
            Dataset instance.
        """
        x, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=num_features,
            n_redundant=0,
            n_repeated=0,
            n_classes=num_classes,
            random_state=42,
        )

        x, y = x.astype("float32"), y.astype("int32")
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

        feature_names: t.List[str] = [f"feature_{i}" for i in range(num_features)]
        type_ = TaskType.BINARY_CLASSIFICATION if num_classes == 2 else TaskType.MULTI_CLASS_CLASSIFICATION
        return Dataset(
            metadata=DatasetMetadata(
                name="random-classification-dataset",
                version="default",
                task=ClassificationTask(type_=type_, num_classes=num_classes),
                features=[Feature(name, FeatureType.CONTINUOUS) for name in feature_names],
            ),
            splits={
                "train": DatasetSplit(x=pd.DataFrame(x_train, columns=feature_names), y=pd.Series(y_train)),
                "test": DatasetSplit(x=pd.DataFrame(x_test, columns=feature_names), y=pd.Series(y_test)),
            },
        )

    @staticmethod
    def make_regression_dataset(num_samples: int = 500, num_features: int = 20) -> Dataset:
        """Create random regression datasets.

        Args:
            num_samples: Dataset size (total across train/test split).
            num_features: Number of features.

        Returns:
            Dataset instance.
        """
        x, y = make_regression(
            n_samples=num_samples, n_features=num_features, n_informative=num_features, random_state=42
        )

        x, y = x.astype("float32"), y.astype("int32")
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

        feature_names: t.List[str] = [f"feature_{i}" for i in range(num_features)]
        return Dataset(
            metadata=DatasetMetadata(
                name="random-regression-dataset",
                version="default",
                task=RegressionTask(),
                features=[Feature(name, FeatureType.CONTINUOUS) for name in feature_names],
            ),
            splits={
                "train": DatasetSplit(x=pd.DataFrame(x_train, columns=feature_names), y=pd.Series(y_train)),
                "test": DatasetSplit(x=pd.DataFrame(x_test, columns=feature_names), y=pd.Series(y_test)),
            },
        )

    def setUp(self) -> None:
        self.classification_dataset = self.make_classification_dataset()
        self.regression_dataset = self.make_regression_dataset()

    def _test_manual_loading(
        self, estimator_cls: t.Type[Estimator], model_cls: t.Type, dataset: Dataset, params: t.Optional[t.Dict] = None
    ) -> None:
        if params is None:
            params = {"n_estimators": 5}
        estimator = estimator_cls(params, dataset.metadata)
        estimator.fit_model(dataset)
        with TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            estimator.save_model(save_dir)
            model = Model.load_model(
                save_dir, LegacySavedModelInfo(getattr(estimator_cls, "NAME"), dataset.metadata.task.type.value)
            )
            self.assertIsInstance(model, model_cls)

    def _test_automated_loading(
        self, estimator_cls: t.Type[Estimator], model_cls: t.Type, dataset: Dataset, params: t.Optional[t.Dict] = None
    ) -> None:
        if params is None:
            params = {"n_estimators": 5}

        work_dir = os.getcwd()
        with TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            estimator_cls.fit(
                params,
                Context(
                    metadata=Metadata(
                        dataset=dataset.metadata.name, model=getattr(estimator_cls, "NAME"), run_type=RunType.TRAIN
                    ),
                    dataset=dataset,
                ),
            )

            model = Model.load_model(Path(temp_dir))
            self.assertIsInstance(model, model_cls)

        os.chdir(work_dir)

    def test_catboost_manual_loading(self) -> None:
        self._test_manual_loading(CatboostEstimator, CatBoostClassifier, self.classification_dataset)
        self._test_manual_loading(CatboostEstimator, CatBoostRegressor, self.regression_dataset)

    def test_catboost_automated_loading(self) -> None:
        self._test_automated_loading(CatboostEstimator, CatBoostClassifier, self.classification_dataset)
        self._test_automated_loading(CatboostEstimator, CatBoostRegressor, self.regression_dataset)

    def test_xgboost_manual_loading(self) -> None:
        self._test_manual_loading(XGBoostEstimator, XGBClassifier, self.classification_dataset)
        self._test_manual_loading(XGBoostEstimator, XGBRegressor, self.regression_dataset)

    def test_xgboost_automated_loading(self) -> None:
        self._test_automated_loading(XGBoostEstimator, XGBClassifier, self.classification_dataset)
        self._test_automated_loading(XGBoostEstimator, XGBRegressor, self.regression_dataset)

    def test_lightgbm_manual_loading(self) -> None:
        self._test_manual_loading(LightGBMClassifierEstimator, Booster, self.classification_dataset)
        self._test_manual_loading(LightGBMClassifierEstimator, Booster, self.regression_dataset)

    def test_lightgbm_automated_loading(self) -> None:
        self._test_automated_loading(LightGBMClassifierEstimator, Booster, self.classification_dataset)
        self._test_automated_loading(LightGBMClassifierEstimator, Booster, self.regression_dataset)

    def test_scikit_learn_manual_loading(self) -> None:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        self._test_manual_loading(DummyEstimator, DummyClassifier, self.classification_dataset, {})
        self._test_manual_loading(DummyEstimator, DummyRegressor, self.regression_dataset, {})

        self._test_manual_loading(RandomForestEstimator, RandomForestClassifier, self.classification_dataset)
        self._test_manual_loading(RandomForestEstimator, RandomForestRegressor, self.regression_dataset)

    def test_scikit_learn_automated_loading(self) -> None:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        self._test_automated_loading(DummyEstimator, DummyClassifier, self.classification_dataset, {})
        self._test_automated_loading(DummyEstimator, DummyRegressor, self.regression_dataset, {})

        self._test_automated_loading(RandomForestEstimator, RandomForestClassifier, self.classification_dataset)
        self._test_automated_loading(RandomForestEstimator, RandomForestRegressor, self.regression_dataset)

    def test_rapids_random_forest_manual_loading(self) -> None:
        try:
            from cuml.ensemble import RandomForestClassifier, RandomForestRegressor

            from xtime.estimators._rapids import RandomForestEstimator

            self._test_manual_loading(RandomForestEstimator, RandomForestClassifier, self.classification_dataset)
            self._test_manual_loading(RandomForestEstimator, RandomForestRegressor, self.regression_dataset)
        except ImportError:
            return

    def test_rapids_random_forest_automated_loading(self) -> None:
        try:
            from cuml.ensemble import RandomForestClassifier, RandomForestRegressor

            from xtime.estimators._rapids import RandomForestEstimator

            self._test_automated_loading(RandomForestEstimator, RandomForestClassifier, self.classification_dataset)
            self._test_automated_loading(RandomForestEstimator, RandomForestRegressor, self.regression_dataset)
        except ImportError:
            return
