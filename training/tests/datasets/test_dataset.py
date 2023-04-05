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
from pathlib import Path
from unittest import TestCase

from xtime.contrib.unittest_ext import with_temp_work_dir
from xtime.datasets.dataset import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit, build_dataset


class TestDataset(TestCase):
    @with_temp_work_dir
    def test_save_load(self) -> None:
        ds: Dataset = build_dataset("churn_modelling:default")
        self.assertEqual(ds.metadata.name, "churn_modelling")
        self.assertEqual(ds.metadata.version, "default")

        ds.save(Path.cwd())
        loaded_ds: Dataset = Dataset.load(Path.cwd())

        self.assertDictEqual(ds.metadata.to_json(), loaded_ds.metadata.to_json())
        self.assertEqual(sorted(list(ds.splits.keys())), sorted(list(loaded_ds.splits.keys())))

        # TODO: add unit tests for features and targets
