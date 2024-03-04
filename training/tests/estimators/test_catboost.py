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

import pytest

from xtime.contrib.unittest_ext import with_temp_work_dir
from xtime.datasets import Dataset
from xtime.estimators._catboost import CatboostEstimator
from xtime.estimators.estimator import unit_test_check_metrics, unit_test_train_model

pytestmark = pytest.mark.estimators


class TestCatboostEstimator(TestCase):
    @with_temp_work_dir
    def test_churn_modelling_numerical(self) -> None:
        ds: Dataset = Dataset.create("churn_modelling:numerical")
        self.assertIsInstance(ds, Dataset)

        metrics: t.Dict = unit_test_train_model(self, "catboost", CatboostEstimator, ds)
        unit_test_check_metrics(self, ds.metadata.task, metrics)

    @with_temp_work_dir
    def test_year_prediction_msd_default(self) -> None:
        ds: Dataset = Dataset.create("year_prediction_msd:default")
        self.assertIsInstance(ds, Dataset)

        metrics: t.Dict = unit_test_train_model(self, "catboost", CatboostEstimator, ds)
        unit_test_check_metrics(self, ds.metadata.task, metrics)
