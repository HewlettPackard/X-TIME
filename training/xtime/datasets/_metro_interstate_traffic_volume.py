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
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from xtime.datasets import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from xtime.datasets.preprocessing import TimeSeries, TimeSeriesEncoderV1
from xtime.errors import DatasetError
from xtime.ml import Feature, FeatureType, RegressionTask, TaskType

__all__ = ["MITVBuilder"]

logger = logging.getLogger(__name__)


_XTIME_DATASETS_MITV = "XTIME_DATASETS_MITV"
"""Environment variable that points to a directory with MITV dataset."""

_MITV_HOME_PAGE = "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
"""Dataset home page."""

_MITV_DATASET_FILE = "Metro_Interstate_Traffic_Volume.csv"
"""File containing raw (unprocessed) MITV dataset that is located inside _XTIME_DATASETS_MITV directory."""


class MITVBuilder(DatasetBuilder):
    """MITV: Metro Interstate Traffic Volume.

    AHourly Minneapolis-St Paul, MN traffic volume for westbound I-94. Includes weather and
    holiday features from 2012-2018. Hourly Interstate 94 Westbound traffic volume for MN
    DoT ATR station 301, roughly midway between Minneapolis and St Paul, MN. Hourly weather
    features and holidays included for impacts on traffic volume.:
        https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume
    """

    NAME = "metro_interstate_traffic_volume"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset)
        self.encoder = TimeSeriesEncoderV1()

    def _check_pre_requisites(self) -> None:
        # Check raw dataset exists.
        if _XTIME_DATASETS_MITV not in os.environ:
            raise DatasetError.missing_prerequisites(
                f"No environment variable found ({_XTIME_DATASETS_MITV}) that should point to a directory with "
                f"MITV (Metro Interstate Traffic Volume) dataset that can be downloaded from `{_MITV_HOME_PAGE}`."
            )
        self._dataset_dir = Path(os.environ[_XTIME_DATASETS_MITV]).absolute()
        if self._dataset_dir.is_file():
            self._dataset_dir = self._dataset_dir.parent
        if not (self._dataset_dir / _MITV_DATASET_FILE).is_file():
            raise DatasetError.missing_prerequisites(
                f"MITV dataset location was identified as `{self._dataset_dir}`, but this is either not a directory "
                f"or dataset file (`{_MITV_DATASET_FILE}`) not found in this location. Please, download v1.1 of this "
                f"dataset from its home page `{_MITV_HOME_PAGE}`."
            )

        # Check `tsfresh` library can be imported.
        try:
            import tsfresh.feature_extraction.feature_calculators as ts_features

        except ImportError:
            raise DatasetError.missing_prerequisites(
                f"The MITV dataset requires `tsfresh` library to compute ML features. If it has not been installed, "
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

        train_df = pd.read_csv(self._dataset_dir / (_MITV_DATASET_FILE[0:-4] + "-default-train.csv"))
        test_df = pd.read_csv(self._dataset_dir / (_MITV_DATASET_FILE[0:-4] + "-default-test.csv"))

        features = [
            Feature("holiday", FeatureType.ORDINAL),
            Feature("temp", FeatureType.CONTINUOUS),
            Feature("rain_1h", FeatureType.CONTINUOUS),
            Feature("snow_1h", FeatureType.CONTINUOUS),
            Feature("clouds_all", FeatureType.ORDINAL),
            Feature("weather_main", FeatureType.ORDINAL),
            Feature("weather_description", FeatureType.ORDINAL),
            Feature("year", FeatureType.ORDINAL),
            Feature("month", FeatureType.ORDINAL),
            Feature("day", FeatureType.ORDINAL),
            Feature("hour", FeatureType.ORDINAL),
        ]

        # Check that data frames contains expected columns (11 is for features, 1 is for 1traffic_volume1).
        assert train_df.shape[1] == len(features) + 1, "Train data frame contains wrong number of columns."
        assert test_df.shape[1] == len(features) + 1, "Test data frame contains wrong number of columns."
        for feature in features:
            assert (
                feature.name in train_df.columns
            ), f"Missing column `{feature}` in train dataframe (columns={list(train_df.columns)})."
            assert (
                feature.name in test_df.columns
            ), f"Missing column `{feature}` in test dataframe (columns={list(train_df.columns)})."

        label: str = "traffic_volume"

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=MITVBuilder.NAME,
                version="default",
                task=RegressionTask(),
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
        """Clean raw MITV dataset."""
        # Do not clean it again if it has already been cleaned.
        # Dataset provides a single file in `Metro_Interstate_Traffic_Volume.csv` format
        # Use the raw file directly
        _clean_dataset_file = (self._dataset_dir / _MITV_DATASET_FILE).with_suffix(".csv")
        if _clean_dataset_file.is_file():
            return

    def _create_default_dataset(self) -> None:
        """Create default train/test splits and save them to files.

        Input to this function is the clean dataset created by the `_clean_dataset` method of this class.
        """
        # Do not generate datasets if they have already been generated.
        default_train_dataset_file = self._dataset_dir / (_MITV_DATASET_FILE[0:-4] + "-default-train.csv")
        default_test_dataset_file = self._dataset_dir / (_MITV_DATASET_FILE[0:-4] + "-default-test.csv")
        if default_train_dataset_file.is_file() and default_test_dataset_file.is_file():
            return

        # Load clean dataset into a data frame
        # holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,weather_description,date_time,traffic_volume
        clean_dataset_file = (self._dataset_dir / _MITV_DATASET_FILE).with_suffix(".csv")
        assert clean_dataset_file.is_file(), f"Clean dataset does not exist (this is internal error)."
        dtypes = {
            "holiday": "string",
            "temp": "float64",
            "rain_1h": "float64",
            "snow_1h": "float64",
            "clouds_all": "int64",
            "weather_main": "string",
            "weather_description": "string",
            "date_time": "string",
            "traffic_volume": "int64",
        }
        df: pd.DataFrame = pd.read_csv(clean_dataset_file, dtype=dtypes)

        # Sanity check: Dataset host link mentions no missing values
        assert not df.isna().any().any(), "There are missing values in the DataFrame"

        assert df.shape[1] == 9, f"Clean dataset expected to have 9 columns (shape={df.shape})."
        for col in dtypes.keys():
            assert col in df.columns, f"Clean dataset does not provide `{col}` column."

        # Encode labels (that are strings here) into numerical representation
        label_encoder = LabelEncoder().fit(df["holiday"])
        df["holiday"] = label_encoder.transform(df["holiday"])

        label_encoder = LabelEncoder().fit(df["weather_main"])
        df["weather_main"] = label_encoder.transform(df["weather_main"])

        label_encoder = LabelEncoder().fit(df["weather_description"])
        df["weather_description"] = label_encoder.transform(df["weather_description"])

        # Separate columns for: year, month, day, hour from `df[date_time]`
        df["year"] = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S").year for i in df["date_time"]]
        df["month"] = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S").month for i in df["date_time"]]
        df["day"] = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S").day for i in df["date_time"]]
        df["hour"] = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S").hour for i in df["date_time"]]

        # Don't need `df[date_time]`
        df = df.drop("date_time", axis=1)
        # Covert temp from kelvin to c
        df["temp"] = df["temp"] - 273.15

        assert df.shape[1] == 12, f"Clean dataset expected to have 12 columns (shape={df.shape})."

        # Split train and test dataframes
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

        df_train.to_csv(default_train_dataset_file, index=False)
        df_test.to_csv(default_test_dataset_file, index=False)
