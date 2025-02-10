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
import pickle
import sys
import typing as t
from pathlib import Path
from unittest import TestCase, mock

from xtime.contrib.unittest_ext import with_temp_work_dir
from xtime.datasets.dataset import (
    Dataset,
    DatasetFactory,
    DatasetPrerequisites,
    RegisteredDatasetFactory,
)
from xtime.errors import DatasetError


class TestDataset(TestCase):
    def _check_serialized_dataset_structure(self, directory: Path, splits: t.List[str]) -> None:
        """Check serialized dataset structure.

        Args:
            directory: Path to the directory where dataset was serialized.
            splits: List of expected splits (e.g., ["train", "test"]).
        """
        self.assertTrue((directory / "metadata.yaml").exists(), "Dataset metadata yaml file does not exist.")
        for split_name in splits:
            self.assertTrue(
                (directory / f"{split_name}.pickle").exists(), f"Dataset split ({split_name}) file does not exist."
            )

        pickle_files = list(directory.glob("*.pickle"))
        self.assertEqual(len(splits), len(pickle_files), "Unexpected split files present.")

    @with_temp_work_dir
    def test_save_load(self) -> None:
        ds: Dataset = Dataset.create("churn_modelling:default")
        self.assertEqual(ds.metadata.name, "churn_modelling")
        self.assertEqual(ds.metadata.version, "default")

        cwd: Path = Path.cwd()

        ds.save(cwd)
        self._check_serialized_dataset_structure(cwd, ["train", "test"])

        loaded_ds: Dataset = Dataset.load(cwd)

        self.assertDictEqual(ds.metadata.to_json(), loaded_ds.metadata.to_json())
        self.assertEqual(sorted(list(ds.splits.keys())), sorted(list(loaded_ds.splits.keys())))

        # Check that overwrite parameter works
        with self.assertRaises(FileExistsError):
            ds.save(cwd)  # Default value must be False.
        with self.assertRaises(FileExistsError):
            ds.save(cwd, overwrite=False)  # Explicitly instruct not to overwrite.

        # Overwrite existing dataset
        with open(cwd / "eval.pickle", "wb") as _file:
            pickle.dump({"x": [1, 2, 3], "y": [1, 2, 3]}, _file)
        ds.save(cwd, overwrite=True)  # This should not raise the FileExistsError exception.
        self._check_serialized_dataset_structure(cwd, ["train", "test"])

        # TODO: add unit tests for features and targets

    def test_registered_dataset_factory(self) -> None:
        # Get all registered datasets
        dataset_names: t.List[str] = []
        for dataset_name in RegisteredDatasetFactory.registry.keys():
            for dataset_version in RegisteredDatasetFactory.registry.get(dataset_name)().builders.keys():
                dataset_names.append(f"{dataset_name}:{dataset_version}")
        #
        self.assertTrue(len(dataset_names) > 0, "No registered datasets found.")
        #
        for dataset_name in dataset_names:
            factories: t.List[DatasetFactory] = DatasetFactory.resolve_factories(dataset_name)
            self.assertEqual(1, len(factories), f"Expected one factory for '{dataset_name}' dataset.")
            self.assertIsInstance(
                factories[0],
                RegisteredDatasetFactory,
                f"Expected 'RegisteredDatasetFactory' class for '{dataset_name}' dataset.",
            )

    def test_dataset_prerequisites_get_install_lib_help(self) -> None:
        msg: str = DatasetPrerequisites._get_install_lib_help("openml", "openml")
        self.assertIsInstance(msg, str)

        msg: str = DatasetPrerequisites._get_install_lib_help("openml", ["openml", "datasets", "all"])
        self.assertIsInstance(msg, str)

        msg: str = DatasetPrerequisites._get_install_lib_help("tsfresh", ["timeseries", "datasets", "all"])
        self.assertIsInstance(msg, str)

    def test_dataset_prerequisites_check_openml(self) -> None:
        with mock.patch.dict(sys.modules, {"openml": None}):
            with self.assertRaises(DatasetError):
                DatasetPrerequisites.check_openml("gas_concentrations", "openml.org/d/1477")

    def test_dataset_prerequisites_check_tsfresh(self) -> None:
        with mock.patch.dict(sys.modules, {"tsfresh": None}):
            with self.assertRaises(DatasetError):
                DatasetPrerequisites.check_tsfresh("fraud_detection")
