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

import logging
import os
import re
import typing as t
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from xtime.datasets import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from xtime.datasets.dataset import DatasetPrerequisites
from xtime.datasets.preprocessing import TimeSeries, TimeSeriesEncoderV1
from xtime.errors import DatasetError
from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

__all__ = ["HARTHBuilder"]

logger = logging.getLogger(__name__)


_XTIME_DATASETS_HARTH = "XTIME_DATASETS_HARTH"
"""Environment variable that points to a directory with HARTH dataset."""

_HARTH_HOME_PAGE = "https://archive.ics.uci.edu/dataset/779/harth"
"""Dataset home page."""

_HARTH_DATASET_FILES = [
    "S006.csv",
    "S008.csv",
    "S009.csv",
    "S010.csv",
    "S012.csv",
    "S013.csv",
    "S014.csv",
    "S015.csv",
    "S016.csv",
    "S017.csv",
    "S018.csv",
    "S019.csv",
    "S020.csv",
    "S021.csv",
    "S022.csv",
    "S023.csv",
    "S024.csv",
    "S025.csv",
    "S026.csv",
    "S027.csv",
    "S028.csv",
    "S029.csv",
]

_HARTH_DATASET_MERGED_FILE = "harth_merged_data.csv"
"""File containing raw (unprocessed) HARTH dataset that is located inside _XTIME_DATASETS_HARTH directory."""


class HARTHBuilder(DatasetBuilder):
    """HARTH: Human Activity Recognition Trondheim.

    The Human Activity Recognition Trondheim (HARTH) dataset is a professionally-annotated dataset containing
    22 subjects wearing two 3-axial accelerometers for around 2 hours in a free-living setting.
    The sensors were attached to the right thigh and lower back.
    The professional recordings and annotations provide a promising benchmark dataset for researchers to develop
    innovative machine learning approaches for precise HAR in free living.:
        https://archive.ics.uci.edu/dataset/779/harth
    """

    NAME = "harth"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset)
        self.encoder = TimeSeriesEncoderV1()
        self._dataset_dir: t.Optional[Path] = None

    def _check_pre_requisites(self) -> None:
        # Check raw dataset exists.
        if _XTIME_DATASETS_HARTH not in os.environ:
            raise DatasetError.missing_prerequisites(
                f"No environment variable found ({_XTIME_DATASETS_HARTH}) that should point to "
                f"a directory with Human Activity Recognition Trondheim (HARTH) dataset that "
                f"can be downloaded from `{_HARTH_HOME_PAGE}`."
            )
        dataset_dir = Path(os.environ[_XTIME_DATASETS_HARTH]).absolute()
        if dataset_dir.is_file():
            dataset_dir = dataset_dir.parent

        # With multiple files present in this dataset
        # Check for all files and report a list of missing file(s) if not found
        missing_files = []
        for file in _HARTH_DATASET_FILES:
            if not (dataset_dir / file).is_file():
                missing_files.append(file)
        if missing_files:
            # Report the missing files
            missing_files_str = ", ".join(missing_files)
            raise DatasetError.missing_prerequisites(
                f"HARTH dataset location was identified as `{dataset_dir}`, but this is either not a directory "
                f"or dataset file(s): {missing_files_str}, not found in this location. Please download the "
                f"dataset from {_HARTH_HOME_PAGE}."
            )
        # Check `tsfresh` library can be imported.
        DatasetPrerequisites.check_tsfresh(self.NAME)
        #
        self._dataset_dir = dataset_dir

    def _build_default_dataset(self, **kwargs) -> Dataset:
        if kwargs:
            raise ValueError(f"{self.__class__.__name__}: `default` dataset does not accept arguments.")
        self._merge_dataset()
        self._create_default_dataset()

        assert self._dataset_dir is not None, "Dataset directory is none."
        dataset_dir: Path = self._dataset_dir

        train_df = pd.read_csv(dataset_dir / (_HARTH_DATASET_MERGED_FILE[0:-4] + "-default-train.csv"))
        test_df = pd.read_csv(dataset_dir / (_HARTH_DATASET_MERGED_FILE[0:-4] + "-default-test.csv"))

        # These are the base feature names (will have prefixes such as):
        # `back_x_`, `back_y_`, `back_z_`, `thigh_x_`, `thigh_y_` and `thigh_z_`
        # This check needs to be consistent with feature generation in `_create_default_dataset` method.
        feature_names = self.encoder.features()
        features = [
            Feature(f"{axis}_{name}", FeatureType.CONTINUOUS)
            for axis, name in product(("back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"), feature_names)
        ]

        # Check that data frames contains expected columns (3 is for three axes, 1 is for label).
        assert train_df.shape[1] == 6 * len(feature_names) + 1, "Train data frame contains wrong number of columns."
        assert test_df.shape[1] == 6 * len(feature_names) + 1, "Test data frame contains wrong number of columns."
        for feature in features:
            assert (
                feature.name in train_df.columns
            ), f"Missing column `{feature}` in train dataframe (columns={list(train_df.columns)})."
            assert (
                feature.name in test_df.columns
            ), f"Missing column `{feature}` in test dataframe (columns={list(train_df.columns)})."

        label: str = "label"

        # Encode labels (that are strings here) into numerical representation (0, num_classes-1).
        label_encoder = LabelEncoder().fit(train_df[label])
        train_df[label] = label_encoder.transform(train_df[label])
        test_df[label] = label_encoder.transform(test_df[label])

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=HARTHBuilder.NAME,
                version="default",
                task=ClassificationTask(TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=12),
                features=features,
                properties={"source": dataset_dir.as_uri()},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(x=train_df.drop(label, axis=1, inplace=False), y=train_df[label]),
                DatasetSplit.TEST: DatasetSplit(x=test_df.drop(label, axis=1, inplace=False), y=test_df[label]),
            },
        )
        return dataset

    def extract_subject_id(self, file_name: str) -> t.Optional[str]:
        """Extract subject ID from a file name.

        Regular expression pattern to match subject ID
        For example: 008 from `S008.csv`, 027 from `S027.csv`
        """
        pattern = r"S(\d+)\.csv"
        match = re.search(pattern, file_name)
        if match:
            return match.group(1)
        else:
            return None

    def _merge_dataset(self) -> None:
        assert self._dataset_dir is not None, "Dataset directory is none."
        dataset_dir: Path = self._dataset_dir

        _merged_dataset_file = dataset_dir / _HARTH_DATASET_MERGED_FILE
        if _merged_dataset_file.is_file():
            return

        subject_data = []

        for file in _HARTH_DATASET_FILES:
            subject_id = self.extract_subject_id(file)  # Extract subject ID from file name
            subject_df = pd.read_csv(dataset_dir / file)
            if subject_df.shape[1] == 9:
                # Check if the DataFrame has an "index" column and drop it if it exists
                # Not all but some of them have "index" column (e.g. `S015.csv`)
                if "index" in subject_df.columns:
                    subject_df.drop(columns=["index"], inplace=True)
                else:
                    # Drop the first column (index 0)
                    # This is a specific case for `S023.csv` file
                    subject_df.drop(subject_df.columns[0], axis=1, inplace=True)
                assert (
                    subject_df.shape[1] == 8
                ), f"Clean dataset expected to have 8 columns after dropping 'index' column (shape={subject_df.shape})."
            else:
                assert subject_df.shape[1] == 8, f"Clean dataset expected to have 8 columns (shape={subject_df.shape})."

            # Original dataset files have the following columns:
            # timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z,label
            # Removed 'index' column from some subject files
            # Add subject ID column
            subject_df["subject_id"] = subject_id
            subject_data.append(subject_df)

        merged_data = pd.concat(subject_data, ignore_index=True)
        merged_data.to_csv(_merged_dataset_file, index=False)

    def _create_default_dataset(self) -> None:
        """Create default train/test splits and save them to files.

        Input to this function is the clean dataset created by the `_clean_dataset` method of this class.
        """
        #
        assert self._dataset_dir is not None, "Dataset directory is none."
        dataset_dir: Path = self._dataset_dir

        # Do not generate datasets if they have already been generated.
        default_train_dataset_file = dataset_dir / (_HARTH_DATASET_MERGED_FILE[0:-4] + "-default-train.csv")
        default_test_dataset_file = dataset_dir / (_HARTH_DATASET_MERGED_FILE[0:-4] + "-default-test.csv")
        if default_train_dataset_file.is_file() and default_test_dataset_file.is_file():
            return

        # Load clean dataset into a data frame (timestamp,back_x,back_y,back_z,thigh_x,thigh_y,thigh_z,label)
        clean_dataset_file = (dataset_dir / _HARTH_DATASET_MERGED_FILE).with_suffix(".csv")
        assert clean_dataset_file.is_file(), "Clean dataset does not exist (this is internal error)."
        dtypes = {
            "timestamp": "object",
            "back_x": "float64",
            "back_y": "float64",
            "back_z": "float64",
            "thigh_x": "float64",
            "thigh_y": "float64",
            "thigh_z": "float64",
            "label": "int64",
            "subject_id": "int64",
        }
        df: pd.DataFrame = pd.read_csv(clean_dataset_file, dtype=dtypes)

        # Sanity check: Dataset host link mentions no missing values
        assert not df.isna().any().any(), "There are missing values in the DataFrame"

        # 8 columns originally and we added a column for subject ID in function: `_merge_dataset`
        assert df.shape[1] == 9, f"Clean dataset expected to have 9 columns (shape={df.shape})."
        for col in dtypes.keys():
            assert col in df.columns, f"Clean dataset does not provide `{col}` column."

        # Split into train/test subsets
        df_train = df[df["subject_id"] <= 23]
        df_test = df[df["subject_id"] > 23]

        # Apply sliding window transformation.
        window_size, stride = 200, 40

        # A list of features required for sliding window
        training_features = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

        # A list of prefixes for `training_features`
        prefixes = ["back_x_", "back_y_", "back_z_", "thigh_x_", "thigh_y_", "thigh_z_"]

        train_windows = TimeSeries.slide(df_train[training_features], None, window_size, stride)
        train_labels = TimeSeries.slide(df_train.label, TimeSeries.mode, window_size, stride)
        test_windows = TimeSeries.slide(df_test[training_features], None, window_size, stride)
        test_labels = TimeSeries.slide(df_test.label, TimeSeries.mode, window_size, stride)

        def _create_dataset(_windows: np.ndarray, _labels: np.ndarray, _file_path: Path) -> None:
            """Convert windows with raw accelerometer values into machine learning features and save to file.
            Args:
                _windows: Rank-3 tensor of shape [NumExamples, WindowSize, NumAxis].
                _labels: Array of labels, number of labels = NumExamples.
                _file_path: File name to write the generated dataset.
            """
            _dataset = pd.DataFrame(self.encoder.encode_many(_windows, prefixes=prefixes))
            _dataset["label"] = _labels.flatten()
            assert _dataset.shape[0] == _windows.shape[0], "error!"
            assert _dataset.shape[1] == 19 * len(prefixes) + 1, "error!"
            _dataset.to_csv(_file_path, index=False)

        _create_dataset(train_windows, train_labels, default_train_dataset_file)
        _create_dataset(test_windows, test_labels, default_test_dataset_file)
