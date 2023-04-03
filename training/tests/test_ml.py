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

import uuid
from unittest import TestCase

from xtime.contrib.unittest_ext import check_enum
from xtime.ml import METRICS, ClassificationTask, Feature, FeatureType, RegressionTask, Task, TaskType, _Metrics


class TestML(TestCase):
    def _check_classification_task(self, task: ClassificationTask, task_type: TaskType, num_classes: int) -> None:
        self.assertIsInstance(task, ClassificationTask)
        self.assertEqual(task.type, task_type)
        self.assertEqual(task.num_classes, num_classes)

        json_obj = task.to_json()
        self.assertEqual(json_obj, {"type": task_type.value, "num_classes": num_classes})

        task = Task.from_json(json_obj)
        self.assertIsInstance(task, ClassificationTask)
        self.assertEqual(task.type, task_type)
        self.assertEqual(task.num_classes, num_classes)

    def _check_feature_type(self, ft: FeatureType, numerical: bool, categorical: bool, nominal: bool) -> None:
        self.assertIsInstance(ft, FeatureType)
        if numerical:
            self.assertTrue(ft.numerical())
        if categorical:
            self.assertTrue(ft.categorical())
        if nominal:
            self.assertTrue(ft.nominal())

    def _check_feature(self, f: Feature, name: str, ft: FeatureType) -> None:
        self.assertIsInstance(f, Feature)
        self.assertEqual(f.name, name)
        self.assertEqual(f.type, ft)

    def test_run_type(self) -> None:
        check_enum(self, TaskType, TaskType.BINARY_CLASSIFICATION, "BINARY_CLASSIFICATION", "binary_classification")
        check_enum(
            self,
            TaskType,
            TaskType.MULTI_CLASS_CLASSIFICATION,
            "MULTI_CLASS_CLASSIFICATION",
            "multi_class_classification",
        )
        check_enum(self, TaskType, TaskType.REGRESSION, "REGRESSION", "regression")

        self.assertTrue(TaskType.BINARY_CLASSIFICATION.classification())
        self.assertTrue(TaskType.MULTI_CLASS_CLASSIFICATION.classification())
        self.assertTrue(TaskType.REGRESSION.regression())

        self.assertFalse(TaskType.BINARY_CLASSIFICATION.regression())
        self.assertFalse(TaskType.MULTI_CLASS_CLASSIFICATION.regression())
        self.assertFalse(TaskType.REGRESSION.classification())

    def test_classification_task(self) -> None:
        task = ClassificationTask(TaskType.BINARY_CLASSIFICATION)
        self._check_classification_task(task, TaskType.BINARY_CLASSIFICATION, 2)

        task = ClassificationTask(TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=7)
        self._check_classification_task(task, TaskType.MULTI_CLASS_CLASSIFICATION, 7)

    def test_regression_task(self) -> None:
        self.assertEqual(RegressionTask(type_=TaskType.REGRESSION).type, TaskType.REGRESSION)

        task = RegressionTask()
        self.assertEqual(task.type, TaskType.REGRESSION)

        json_obj = task.to_json()
        self.assertEqual(json_obj, {"type": TaskType.REGRESSION.value})

        task = Task.from_json(json_obj)
        self.assertIsInstance(task, RegressionTask)
        self.assertEqual(task.type, TaskType.REGRESSION)

    def test_feature_type(self) -> None:
        check_enum(self, FeatureType, FeatureType.DISCRETE, "DISCRETE", "discrete")
        check_enum(self, FeatureType, FeatureType.CONTINUOUS, "CONTINUOUS", "continuous")
        check_enum(self, FeatureType, FeatureType.ORDINAL, "ORDINAL", "ordinal")
        check_enum(self, FeatureType, FeatureType.NOMINAL, "NOMINAL", "nominal")
        check_enum(self, FeatureType, FeatureType.BINARY, "BINARY", "binary")

        self._check_feature_type(FeatureType.DISCRETE, True, False, False)
        self._check_feature_type(FeatureType.CONTINUOUS, True, False, False)
        self._check_feature_type(FeatureType.ORDINAL, False, True, False)
        self._check_feature_type(FeatureType.NOMINAL, False, True, True)
        self._check_feature_type(FeatureType.BINARY, False, True, True)

    def test_feature(self) -> None:
        for ft in FeatureType:
            name = str(uuid.uuid4()).replace("-", "")
            f = Feature(name, ft)
            self._check_feature(f, name, ft)

            json_obj = f.to_json()
            self.assertDictEqual(json_obj, {"name": name, "type": ft.value})
            self._check_feature(Feature.from_json(json_obj), name, ft)

    def test_metrics(self) -> None:
        for task_type in TaskType:
            metrics = METRICS[task_type]
            self.assertIsInstance(metrics, list)
