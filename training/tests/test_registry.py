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

from xtime.datasets import DatasetBuilder, RegisteredDatasetFactory
from xtime.estimators import Estimator, get_estimator_registry


class TestRegistry(TestCase):
    def check_registry(self, registry, keys: t.List[str], base_cls):
        for key in keys:
            self.assertTrue(issubclass(registry.get(key), base_cls))

    def test_model_registry(self):
        model_registry = get_estimator_registry()
        names = model_registry.keys()

        expected = ["catboost", "dummy", "lightgbm", "rf", "xgboost"]
        self.assertIsInstance(names, list)
        self.assertEqual(len(expected), len(names), msg=f"names={names}")
        self.assertListEqual(expected, sorted(names))
        self.check_registry(model_registry, expected, Estimator)

    def test_dataset_registry(self):
        dataset_registry = RegisteredDatasetFactory.registry
        names = dataset_registry.keys()

        expected = [
            "churn_modelling",
            "eye_movements",
            "forest_cover_type",
            "gas_concentrations",
            "gesture_phase_segmentation",
            "ozone_level_detection_1hr",
            "telco_customer_churn",
            "year_prediction_msd",
            "rossmann_store_sales",
            "wisdm",
            "madeline",
            "fraud_detection",
            "harth",
        ]

        self.assertIsInstance(names, list)
        self.assertEqual(len(expected), len(names), msg=f"names={names}")

        self.assertListEqual(
            sorted(expected), sorted(names), msg=f"expected={sorted(expected)}\nactual={sorted(names)}"
        )
        self.check_registry(dataset_registry, expected, DatasetBuilder)
