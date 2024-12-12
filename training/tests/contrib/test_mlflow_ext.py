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
from unittest import TestCase, mock

from xtime.contrib.mlflow_ext import MLflow


class TestMLflow(TestCase):
    def test_get_run_id(self) -> None:
        run_id = "c2596b2ef44f4d9fa8c9dd62c222abbb"

        self.assertEqual(run_id, MLflow.get_run_id(run_id))
        self.assertEqual(run_id, MLflow.get_run_id(f"mlflow:{run_id}"))
        self.assertEqual(run_id, MLflow.get_run_id(f"mlflow:/{run_id}"))
        self.assertEqual(run_id, MLflow.get_run_id(f"mlflow:///{run_id}"))

    @mock.patch.dict(os.environ)
    def test_get_run_name_none(self) -> None:
        _ = os.environ.pop("MLFLOW_RUN_NAME", None)
        self.assertIsNone(MLflow.get_run_name())

    @mock.patch.dict(os.environ, {"MLFLOW_RUN_NAME": "  run_1645 "})
    def test_get_run_name_run_1645(self) -> None:
        self.assertEqual("run_1645", MLflow.get_run_name())
