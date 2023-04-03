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

from xtime.contrib.unittest_ext import check_enum
from xtime.run import Metadata, RunType


class TestRun(TestCase):
    def test_run_type(self) -> None:
        check_enum(self, RunType, RunType.HPO, "HPO", "hpo")
        check_enum(self, RunType, RunType.TRAIN, "TRAIN", "train")

    def test_metadata(self) -> None:
        md = Metadata(dataset="imagenet", model="resnet50", run_type=RunType.TRAIN)

        self.assertEqual(md.dataset, "imagenet")
        self.assertEqual(md.model, "resnet50")
        self.assertEqual(md.run_type, RunType.TRAIN)
        self.assertDictEqual(md.fit_params, {})

        self.assertDictEqual(
            md.to_json(), {"dataset": "imagenet", "model": "resnet50", "run_type": "train", "fit_params": {}}
        )
