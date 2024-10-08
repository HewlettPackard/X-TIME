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
import typing as t
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from xtime.datasets import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from xtime.datasets.dataset import DatasetPrerequisites
from xtime.datasets.preprocessing import TimeSeriesEncoderV1
from xtime.errors import DatasetError
from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

__all__ = ["MADELINEBuilder"]

logger = logging.getLogger(__name__)


_XTIME_DATASETS_MADELINE = "XTIME_DATASETS_MADELINE"
"""Environment variable that points to a directory with MADELINE dataset."""

_MADELINE_HOME_PAGE = "https://www.openml.org/search?type=data&sort=runs&id=41144&status=active"
"""Dataset home page."""

_MADELINE_DATASET_FILE = "file79ff4c183a4e.arff"
"""File containing raw (unprocessed) MADELINE dataset that is located inside _XTIME_DATASETS_MADELINE directory."""


class MADELINEBuilder(DatasetBuilder):
    """MADELINE: 4Paradigm dataset.

    The goal of this challenge is to expose the research community to real world datasets of interest to 4Paradigm:
        https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=madeline&id=41144
    """

    NAME = "madeline"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset)
        self.encoder = TimeSeriesEncoderV1()
        self._dataset_dir: t.Optional[Path] = None

    def _check_pre_requisites(self) -> None:
        # Check raw dataset exists.
        if _XTIME_DATASETS_MADELINE not in os.environ:
            raise DatasetError.missing_prerequisites(
                f"No environment variable found ({_XTIME_DATASETS_MADELINE}) that should point to a directory with "
                f"Madeline dataset that can be downloaded from `{_MADELINE_HOME_PAGE}`."
            )
        dataset_dir = Path(os.environ[_XTIME_DATASETS_MADELINE]).absolute()
        if dataset_dir.is_file():
            dataset_dir = dataset_dir.parent
        if not (dataset_dir / _MADELINE_DATASET_FILE).is_file():
            raise DatasetError.missing_prerequisites(
                f"MADELINE dataset location was identified as `{dataset_dir}`, but this is either not a "
                f"directory or dataset file (`{_MADELINE_DATASET_FILE}`) not found in this location. Please, "
                f"download this dataset from its home page `{_MADELINE_HOME_PAGE}`."
            )
        # Check `tsfresh` library can be imported.
        DatasetPrerequisites.check_tsfresh(self.NAME)
        #
        self._dataset_dir = dataset_dir

    def _build_default_dataset(self, **kwargs) -> Dataset:
        if kwargs:
            raise ValueError(f"{self.__class__.__name__}: `default` dataset does not accept arguments.")
        self._clean_dataset()
        self._create_default_dataset()

        assert self._dataset_dir is not None, "Dataset directory is none."
        dataset_dir: Path = self._dataset_dir

        train_df = pd.read_csv(dataset_dir / (_MADELINE_DATASET_FILE[0:-5] + "-default-train.csv"))
        test_df = pd.read_csv(dataset_dir / (_MADELINE_DATASET_FILE[0:-5] + "-default-test.csv"))

        # Features values are Integers
        features = [
            Feature(col, FeatureType.ORDINAL, cardinality=int(train_df[col].nunique()))
            for col in train_df.columns
            if col != "label"
        ]

        # Check that data frames contains expected columns (259 Features: V1 to V259, 1 is for label).
        assert (
            train_df.shape[1] == len(features) + 1
        ), f"Train data frame contains wrong number of columns (shape={train_df.shape})."
        assert (
            test_df.shape[1] == len(features) + 1
        ), f"Test data frame contains wrong number of columns (shape={test_df.shape})."

        label: str = "label"

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=MADELINEBuilder.NAME,
                version="default",
                task=ClassificationTask(TaskType.BINARY_CLASSIFICATION, num_classes=2),
                features=features,
                properties={"source": dataset_dir.as_uri()},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(x=train_df.drop(label, axis=1, inplace=False), y=train_df[label]),
                DatasetSplit.TEST: DatasetSplit(x=test_df.drop(label, axis=1, inplace=False), y=test_df[label]),
            },
        )
        return dataset

    def _clean_dataset(self) -> None:
        """Clean raw MADELINE dataset."""
        assert self._dataset_dir is not None, "Dataset directory is none."
        dataset_dir: Path = self._dataset_dir

        # Do not clean it again if it has already been cleaned.
        _clean_dataset_file = (dataset_dir / _MADELINE_DATASET_FILE).with_suffix(".csv")
        if _clean_dataset_file.is_file():
            return

        with open(dataset_dir / _MADELINE_DATASET_FILE, "rt") as input_stream:
            with open(_clean_dataset_file, "wt") as output_stream:
                # As shown in raw file at the top
                # Write labels (label), and features (V1 to V259)
                output_stream.write("label," + ",".join(f"V{i}" for i in range(1, 260)) + "\n")
                for idx, line in enumerate(input_stream):
                    # Skip lines starting with "@"
                    if line.startswith("@"):
                        continue
                    # Strip double quotes from label column in every line
                    elif line.startswith('"'):
                        line = line.replace('"', "")
                        output_stream.write(line)

    def _create_default_dataset(self) -> None:
        """Create default train/test splits and save them to files.

        Input to this function is the clean dataset created by the `_clean_dataset` method of this class.
        """
        #
        assert self._dataset_dir is not None, "Dataset directory is none."
        dataset_dir: Path = self._dataset_dir

        # Do not generate datasets if they have already been generated.
        default_train_dataset_file = dataset_dir / (_MADELINE_DATASET_FILE[0:-5] + "-default-train.csv")
        default_test_dataset_file = dataset_dir / (_MADELINE_DATASET_FILE[0:-5] + "-default-test.csv")
        if default_train_dataset_file.is_file() and default_test_dataset_file.is_file():
            return

        clean_dataset_file = (dataset_dir / _MADELINE_DATASET_FILE).with_suffix(".csv")
        assert clean_dataset_file.is_file(), "Clean dataset does not exist (this is internal error)."

        df: pd.DataFrame = pd.read_csv(clean_dataset_file)

        # Check for missing values
        assert not df.isna().any().any(), "There are missing values in the DataFrame"

        # Raw file has 260 columns (259 features, labels)
        assert df.shape[1] == 260, f"Clean dataset expected to have 260 columns (shape={df.shape})."

        # Split train and test dataframes
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

        df_train.to_csv(default_train_dataset_file, index=False)
        df_test.to_csv(default_test_dataset_file, index=False)
