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
import typing as t
from unittest import TestCase

from xtime.datasets import Dataset, DatasetMetadata
from xtime.hparams import get_hparams
from xtime.ml import ClassificationTask, RegressionTask, Task, TaskType
from xtime.run import Context, Metadata, RunType

TaskLike = t.TypeVar("TaskLike", bound=Task)


class TestRecommender(TestCase):
    @staticmethod
    def _get_context(model: str, task: TaskLike) -> Context:
        return Context(
            metadata=Metadata(dataset="iris", model=model, run_type=RunType.TRAIN),
            dataset=Dataset(metadata=DatasetMetadata(name="iris", version="0.0.1", task=task)),
        )

    def test_default_lightgbm(self):
        params: t.Dict = get_hparams(
            source="default",
            ctx=self._get_context(
                "lightgbm", ClassificationTask(type_=TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=3)
            ),
        )
        self.assertIsInstance(params, dict)
        self.assertDictEqual(
            params,
            {
                "n_estimators": 100,
                "learning_rate": 0.3,
                "max_depth": 6,
                "colsample_bytree": 1,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "random_state": 1,
            },
        )

    def test_default_dummy_classifier(self):
        params: t.Dict = get_hparams(
            source="default",
            ctx=self._get_context(
                "dummy", ClassificationTask(type_=TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=3)
            ),
        )
        self.assertIsInstance(params, dict)
        self.assertDictEqual(params, {"strategy": "prior", "random_state": 1})

    def test_default_dummy_regressor(self):
        params: t.Dict = get_hparams(source="default", ctx=self._get_context("dummy", RegressionTask()))
        self.assertIsInstance(params, dict)
        self.assertDictEqual(params, {"strategy": "mean"})

    def test_default_rf(self):
        params: t.Dict = get_hparams(
            source="default",
            ctx=self._get_context("rf", ClassificationTask(type_=TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=3)),
        )
        self.assertIsInstance(params, dict)
        self.assertDictEqual(params, {"n_estimators": 100, "max_depth": 6, "random_state": 1})

    def test_default_catboost(self):
        params: t.Dict = get_hparams(
            source="default",
            ctx=self._get_context(
                "catboost", ClassificationTask(type_=TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=3)
            ),
        )
        self.assertIsInstance(params, dict)
        self.assertDictEqual(
            params,
            {
                "learning_rate": 0.03,
                "random_strength": 1,
                "depth": 6,
                "l2_leaf_reg": 3,
                "bagging_temperature": 1,
                "leaf_estimation_iterations": 1,
                "random_state": 1,
            },
        )

    def test_default_xgboost(self):
        params: t.Dict = get_hparams(
            source="default",
            ctx=self._get_context(
                "xgboost", ClassificationTask(type_=TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=3)
            ),
        )
        self.assertIsInstance(params, dict)
        self.assertDictEqual(
            params,
            {
                "n_estimators": 100,
                "learning_rate": 0.3,
                "max_depth": 6,
                "subsample": 1,
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "min_child_weight": 1,
                "reg_alpha": 0,
                "reg_lambda": 1,
                "gamma": 0,
                "random_state": 1,
            },
        )
