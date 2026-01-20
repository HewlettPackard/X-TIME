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

__all__ = ["AEPBuilder"]

logger = logging.getLogger(__name__)


_XTIME_DATASETS_AEP = "XTIME_DATASETS_AEP"
"""Environment variable that points to a directory with AEP dataset."""

_AEP_HOME_PAGE = "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction"
"""Dataset home page."""

_AEP_DATASET_FILE = "energydata_complete.csv"
"""File containing raw (unprocessed) AEP dataset that is located inside _XTIME_DATASETS_AEP directory."""


class AEPBuilder(DatasetBuilder):
    """AEP: Appliances Energy Prediction.

    Experimental data used to create regression models of appliances energy use in a low energy building.
    The data set is at 10 min for about 4.5 months. The house temperature and humidity conditions were
    monitored with a ZigBee wireless sensor network. Each wireless node transmitted the temperature and
    humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods.
    The energy data was logged every 10 minutes with m-bus energy meters. 
        https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
    """

    NAME = "appliances_energy_prediction"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset)
        self.encoder = TimeSeriesEncoderV1()

    def _check_pre_requisites(self) -> None:
        # Check raw dataset exists.
        if _XTIME_DATASETS_AEP not in os.environ:
            raise DatasetError.missing_prerequisites(
                f"No environment variable found `{_XTIME_DATASETS_AEP}` that should point to a directory with "
                f"AEP (Appliances Energy Prediction) dataset `{_AEP_DATASET_FILE}` that can be"
                f"downloaded from `{_AEP_HOME_PAGE}`."
            )
        self._dataset_dir = Path(os.environ[_XTIME_DATASETS_AEP]).absolute()
        if self._dataset_dir.is_file():
            self._dataset_dir = self._dataset_dir.parent
        if not (self._dataset_dir / _AEP_DATASET_FILE).is_file():
            raise DatasetError.missing_prerequisites(
                f"AEP (Appliances Energy Prediction) dataset location was identified as `{self._dataset_dir}`, but this is "
                f"either not a directory or dataset file (`{_AEP_DATASET_FILE}`) not found in this location. "
                f" Please, download `{_AEP_DATASET_FILE}` of this dataset from its home page `{_AEP_HOME_PAGE}`."
            )

        # Check `tsfresh` library can be imported.
        try:
            import tsfresh.feature_extraction.feature_calculators as ts_features

        except ImportError:
            raise DatasetError.missing_prerequisites(
                f"The AEP dataset requires `tsfresh` library to compute ML features. If it has not been installed, "
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

        train_df = pd.read_csv(self._dataset_dir / (_AEP_DATASET_FILE[0:-4] + "-default-train.csv"))
        test_df = pd.read_csv(self._dataset_dir / (_AEP_DATASET_FILE[0:-4] + "-default-test.csv"))
        
        _ordinal_features = ["year", "month", "day", "hour", "min"]
        _drop_features = ["Appliances"]
        
        features = [] 
        for feature in train_df.columns:
            if feature in _drop_features:
                continue
            if feature in _ordinal_features:
                features.append(Feature(feature, FeatureType.ORDINAL))
            else:
                features.append(Feature(feature, FeatureType.CONTINUOUS))
        
        # Check that data frames contains expected columns
        assert train_df.shape[1] == len(features) + 1, "Train data frame contains wrong number of columns."
        assert test_df.shape[1] == len(features) + 1, "Test data frame contains wrong number of columns."
        for feature in features:
            assert (
                feature.name in train_df.columns
            ), f"Missing column `{feature}` in train dataframe (columns={list(train_df.columns)})."
            assert (
                feature.name in test_df.columns
            ), f"Missing column `{feature}` in test dataframe (columns={list(train_df.columns)})."

        target: str = "Appliances"

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=AEPBuilder.NAME,
                version="default",
                task=RegressionTask(TaskType.REGRESSION),
                features=features,
                properties={"source": self._dataset_dir.as_uri()},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(x=train_df.drop(target, axis=1, inplace=False), y=train_df[target]),
                DatasetSplit.TEST: DatasetSplit(x=test_df.drop(target, axis=1, inplace=False), y=test_df[target]),
            },
        )
        return dataset

    def _clean_dataset(self) -> None:
        """Clean raw AEP dataset."""
        # Do not clean it again if it has already been cleaned.
        # Dataset provides a single file in `.csv` format with missing values
        # We will use the raw file directly
        _clean_dataset_file = (self._dataset_dir / _AEP_DATASET_FILE).with_suffix(".csv")
        if _clean_dataset_file.is_file():
            return

    def _create_default_dataset(self) -> None:
        """Create default train/test splits and save them to files.

        Input to this function is the clean dataset created by the `_clean_dataset` method of this class.
        """
        # Do not generate datasets if they have already been generated.
        default_train_dataset_file = self._dataset_dir / (_AEP_DATASET_FILE[0:-4] + "-default-train.csv")
        default_test_dataset_file = self._dataset_dir / (_AEP_DATASET_FILE[0:-4] + "-default-test.csv")
        if default_train_dataset_file.is_file() and default_test_dataset_file.is_file():
            return

        # Load clean dataset into a data frame (No,year,month,day,hour,pm2.5,DEWP,TEMP,PRES,cbwd,Iws,Is,Ir)
        clean_dataset_file = (self._dataset_dir / _AEP_DATASET_FILE).with_suffix(".csv")
        assert clean_dataset_file.is_file(), f"Clean dataset does not exist (this is internal error)."

        df: pd.DataFrame = pd.read_csv(clean_dataset_file)
        # print(f' --- Reading the {clean_dataset_file} file --- ')
        print(df.head())
        print(df.dtypes)

        # Sanity check for missing values
        assert not df.isna().any().any(), "There are missing values in the DataFrame"
        
        assert df.shape[1] == 29, f"Clean dataset expected to have 29 columns (shape={df.shape})."
        
        # Separate columns for: year, month, day, hour from `df[date]`
        
        # convert the date column into a datetime object
        df['date'] = pd.to_datetime(df['date'])
        
        # extract the day, month, and year components
        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['hour'] = df['date'].dt.hour
        df['min'] = df['date'].dt.minute
        
        # show the modified data frame
        print(df)
        
        # Don't need `df[date]`
        df = df.drop("date", axis=1)
        
        assert df.shape[1] == 33, f"Clean dataset expected to have 33 columns (shape={df.shape})."
        print(df.head())
        # Split train and test dataframes
        df_train, df_test = train_test_split(df, test_size=0.2)

        df_train.to_csv(default_train_dataset_file, index=False)
        df_test.to_csv(default_test_dataset_file, index=False)
