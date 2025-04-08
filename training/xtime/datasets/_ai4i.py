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
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from xtime.datasets import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from xtime.datasets.preprocessing import TimeSeriesEncoderV1
from xtime.errors import DatasetError
from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

__all__ = ["AI4IBuilder"]

logger = logging.getLogger(__name__)


_XTIME_DATASETS_AI4I = "XTIME_DATASETS_AI4I"
"""Environment variable that points to a directory with AI4I dataset."""

_AI4I_HOME_PAGE = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
"""Dataset home page."""

_AI4I_DATASET_FILE = "ai4i2020.csv"
"""File containing raw (unprocessed) AI4I dataset that is located inside _XTIME_DATASETS_AI4I directory."""


class AI4IBuilder(DatasetBuilder):
    """AI4I 2020 Predictive Maintenance Dataset.

    The AI4I 2020 Predictive Maintenance Dataset is a synthetic dataset that reflects real predictive
    maintenance data encountered in industry:
        https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
    """

    NAME = "ai4i"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset)
        self.encoder = TimeSeriesEncoderV1()

    def _check_pre_requisites(self) -> None:
        # Check raw dataset exists.
        if _XTIME_DATASETS_AI4I not in os.environ:
            raise DatasetError.missing_prerequisites(
                f"No environment variable found ({_XTIME_DATASETS_AI4I}) that should point to a directory with "
                f"AI4I dataset that can be downloaded from `{_AI4I_HOME_PAGE}`."
            )
        self._dataset_dir = Path(os.environ[_XTIME_DATASETS_AI4I]).absolute()
        if self._dataset_dir.is_file():
            self._dataset_dir = self._dataset_dir.parent
        if not (self._dataset_dir / _AI4I_DATASET_FILE).is_file():
            raise DatasetError.missing_prerequisites(
                f"AI4I dataset location was identified as `{self._dataset_dir}`, but this is either not a "
                f"directory or dataset file (`{_AI4I_DATASET_FILE}`) not found in this location. Please, "
                f"download this dataset from its home page `{_AI4I_HOME_PAGE}`."
            )

        # Check `tsfresh` library can be imported.
        try:
            import tsfresh.feature_extraction.feature_calculators as ts_features

        except ImportError:
            raise DatasetError.missing_prerequisites(
                "The AI4I dataset requires `tsfresh` library to compute ML features. If it has not been installed, "
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

        train_df = pd.read_csv(self._dataset_dir / (_AI4I_DATASET_FILE[0:-4] + "-default-train.csv"))
        test_df = pd.read_csv(self._dataset_dir / (_AI4I_DATASET_FILE[0:-4] + "-default-test.csv"))

        # These are the features this dataset provides after cleaning up
        features = [
            Feature("type", FeatureType.NOMINAL),
            Feature("air_temp_K", FeatureType.CONTINUOUS),
            Feature("process_temp_K", FeatureType.CONTINUOUS),
            Feature("rotational_speed_rpm", FeatureType.ORDINAL),
            Feature("torque_Nm", FeatureType.CONTINUOUS),
            Feature("tool_wear_min", FeatureType.ORDINAL),
        ]

        assert (
            train_df.shape[1] == len(features) + 1
        ), f"Train data frame contains wrong number of columns (shape={train_df.shape})."
        assert (
            test_df.shape[1] == len(features) + 1
        ), f"Test data frame contains wrong number of columns (shape={test_df.shape})."

        label: str = "machine_failure"

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=AI4IBuilder.NAME,
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
        """Clean raw AI4I dataset."""
        # Do not clean it again if it has already been cleaned.
        # Dataset provides a single file in `ai4i2020.csv` format
        # Use the raw file directly
        _clean_dataset_file = (self._dataset_dir / _AI4I_DATASET_FILE).with_suffix(".csv")
        if _clean_dataset_file.is_file():
            return

    def _create_default_dataset(self) -> None:
        """Create default train/test splits and save them to files.

        Input to this function is the clean dataset created by the `_clean_dataset` method of this class.
        """
        # Do not generate datasets if they have already been generated.
        default_train_dataset_file = self._dataset_dir / (_AI4I_DATASET_FILE[0:-4] + "-default-train.csv")
        default_test_dataset_file = self._dataset_dir / (_AI4I_DATASET_FILE[0:-4] + "-default-test.csv")
        if default_train_dataset_file.is_file() and default_test_dataset_file.is_file():
            return

        clean_dataset_file = (self._dataset_dir / _AI4I_DATASET_FILE).with_suffix(".csv")
        assert clean_dataset_file.is_file(), f"Clean dataset does not exist (this is internal error)."

        df: pd.DataFrame = pd.read_csv(clean_dataset_file)

        print(df.head())

        # Sanity check for missing values
        # No missing values on {_AI4I_HOME_PAGE}
        assert not df.isna().any().any(), "There are missing values in the DataFrame"

        # Raw file has 14 columns (8 features, 6 classification labels)
        # Features: `UDI`,`Product ID`,`Type`,
        #           `Air temperature [K]`,`Process temperature [K]`,
        #           `Rotational speed [rpm]`,`Torque [Nm]`,`Tool wear [min]`
        # Classification labels: `Machine failure`,`TWF`,`HDF`,`PWF`,`OSF`,`RNF
        assert df.shape[1] == 14, f"Clean dataset expected to have 14 columns (shape={df.shape})."

        # If at least one of the above failure modes is true, the process fails and
        # the 'machine failure' label is set to 1. It is therefore not transparent to
        # the machine learning method, which of the failure modes has caused the process to fail.
        # For regression, drop `TWF`,`HDF`,`PWF`,`OSF`,`RNF` columns
        # because we want to forecast `Machine failure`.
        df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])
        # print(df.head())

        # Drop `UDI` column
        df = df.drop(df.columns[0], axis=1)
        # print(df.head())
        # print(df.describe())

        # `Product ID` and `Type` provide same information
        # remove `Product ID` and encode `Type` column
        df = df.drop("Product ID", axis=1)
        label_encoder = LabelEncoder()
        df["Type"] = label_encoder.fit_transform(df["Type"])
        print(df.head())

        # rename dataset columns to avoid XGBoost training errors
        # ValueError: feature_names may not contain [, ] or <
        # https://stackoverflow.com/questions/48645846/pythons-xgoost-valueerrorfeature-names-may-not-contain-or
        df.rename(
            columns={
                "Type": "type",
                "Air temperature [K]": "air_temp_K",
                "Process temperature [K]": "process_temp_K",
                "Rotational speed [rpm]": "rotational_speed_rpm",
                "Torque [Nm]": "torque_Nm",
                "Tool wear [min]": "tool_wear_min",
                "Machine failure": "machine_failure",
            },
            inplace=True,
        )
        print(df.head())

        # could be a good representative problem with skewed data
        # 10000 samples['machine_failure']= 0: 9661 and 1: 339
        # only 3.39% have 1 (Machine failure true)
        print(df["machine_failure"].value_counts())

        # Stratified Sampling to make sure df_test has both labels
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=0, stratify=df["machine_failure"])

        # Sanity check for label distribution
        print("train_test_split with stratified sampling for df_train: ", df_train["machine_failure"].value_counts())
        print("train_test_split with stratified sampling for df_test: ", df_test["machine_failure"].value_counts())

        assert df.shape[1] == 7, f"Clean dataset expected to have 7 columns (shape={df.shape})."

        df_train.to_csv(default_train_dataset_file, index=False)
        df_test.to_csv(default_test_dataset_file, index=False)
