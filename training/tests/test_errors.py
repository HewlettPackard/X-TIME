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
from unittest import TestCase

from xtime.errors import ConfigurationError, DatasetError, ErrorCode, EstimatorError, XTimeError


class TestErrors(TestCase):
    def test_errors(self) -> None:
        error = XTimeError("my_custom_error")
        self.assertEqual(ErrorCode.GENERIC_ERROR, error.error_code)

        error = ConfigurationError("my_custom_error")
        self.assertEqual(ErrorCode.CONFIGURATION_ERROR, error.error_code)

        error = EstimatorError("my_custom_error")
        self.assertEqual(ErrorCode.ESTIMATOR_ERROR, error.error_code)

        error = DatasetError("my_custom_error")
        self.assertEqual(ErrorCode.DATASET_ERROR, error.error_code)

    def test_dataset_errors(self) -> None:
        error = DatasetError.missing_prerequisites("missing_prerequisites")
        self.assertIsInstance(error, DatasetError)
        self.assertEqual(ErrorCode.DATASET_MISSING_PREREQUISITES_ERROR, error.error_code)
