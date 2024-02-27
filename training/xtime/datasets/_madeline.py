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
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from xtime.datasets import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from xtime.datasets.preprocessing import TimeSeries, TimeSeriesEncoderV1
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

    def _check_pre_requisites(self) -> None:
        # Check raw dataset exists.
        if _XTIME_DATASETS_MADELINE not in os.environ:
            raise DatasetError.missing_prerequisites(
                f"No environment variable found ({_XTIME_DATASETS_MADELINE}) that should point to a directory with "
                f"Madeline dataset that can be downloaded from `{_MADELINE_HOME_PAGE}`."
            )
        self._dataset_dir = Path(os.environ[_XTIME_DATASETS_MADELINE]).absolute()
        if self._dataset_dir.is_file():
            self._dataset_dir = self._dataset_dir.parent
        if not (self._dataset_dir / _MADELINE_DATASET_FILE).is_file():
            raise DatasetError.missing_prerequisites(
                f"MADELINE dataset location was identified as `{self._dataset_dir}`, but this is either not a directory "
                f"or dataset file (`{_MADELINE_DATASET_FILE}`) not found in this location. Please, download this "
                f"dataset from its home page `{_MADELINE_HOME_PAGE}`."
            )

        # Check `tsfresh` library can be imported.
        try:
            import tsfresh.feature_extraction.feature_calculators as ts_features

        except ImportError:
            raise DatasetError.missing_prerequisites(
                f"The Madeline dataset requires `tsfresh` library to compute ML features. If it has not been installed, "
                "please install it with `pip install tsfresh==0.20.2`. If it is installed, there may be incompatible "
                "CUDA runtime found (see if the cause for the import error is "
                "`numba.cuda.cudadrv.error.NvvmSupportError` exception) - this may occur because `tsfresh` depends on "
                "`stumpy` that depends on `numba` that detects CUDA runtime and tries to use it if available. Try "
                "disabling CUDA for numba by exporting NUMBA_DISABLE_CUDA environment variable "
                "(https://numba.pydata.org/numba-doc/dev/reference/envvars.html#envvar-NUMBA_DISABLE_CUDA): "
                "`export NUMBA_DISABLE_CUDA=1`."
            )

    def _build_default_dataset(self, **kwargs) -> Dataset:
        if kwargs:
            raise ValueError(f"{self.__class__.__name__}: `default` dataset does not accept arguments.")
        self._clean_dataset()
        self._create_default_dataset()

        train_df = pd.read_csv(self._dataset_dir / (_MADELINE_DATASET_FILE[0:-5] + "-default-train.csv"))
        test_df = pd.read_csv(self._dataset_dir / (_MADELINE_DATASET_FILE[0:-5] + "-default-test.csv"))

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
                properties={"source": self._dataset_dir.as_uri()},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(x=train_df.drop(label, axis=1, inplace=False), y=train_df[label]),
                DatasetSplit.TEST: DatasetSplit(x=test_df.drop(label, axis=1, inplace=False), y=test_df[label]),
            },
        )
        return dataset

    def _clean_dataset(self) -> None:
        """Clean raw MADELINE dataset."""
        # Do not clean it again if it has already been cleaned.
        _clean_dataset_file = (self._dataset_dir / _MADELINE_DATASET_FILE).with_suffix(".csv")
        if _clean_dataset_file.is_file():
            return

        with open(self._dataset_dir / _MADELINE_DATASET_FILE, "rt") as input_stream:
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
        # Do not generate datasets if they have already been generated.
        default_train_dataset_file = self._dataset_dir / (_MADELINE_DATASET_FILE[0:-5] + "-default-train.csv")
        default_test_dataset_file = self._dataset_dir / (_MADELINE_DATASET_FILE[0:-5] + "-default-test.csv")
        if default_train_dataset_file.is_file() and default_test_dataset_file.is_file():
            return

        clean_dataset_file = (self._dataset_dir / _MADELINE_DATASET_FILE).with_suffix(".csv")
        assert clean_dataset_file.is_file(), f"Clean dataset does not exist (this is internal error)."

        df: pd.DataFrame = pd.read_csv(clean_dataset_file)

        # Check for missing values
        assert not df.isna().any().any(), "There are missing values in the DataFrame"

        # Raw file has 260 columns (259 features, labels)
        assert df.shape[1] == 260, f"Clean dataset expected to have 260 columns (shape={df.shape})."

        # Split train and test dataframes
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

        df_train.to_csv(default_train_dataset_file, index=False)
        df_test.to_csv(default_test_dataset_file, index=False)
