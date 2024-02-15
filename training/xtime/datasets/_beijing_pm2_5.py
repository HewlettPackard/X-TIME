import logging
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from xtime.datasets import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from xtime.datasets.preprocessing import TimeSeries, TimeSeriesEncoderV1
from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

__all__ = ["BeijingPM25Builder"]

logger = logging.getLogger(__name__)


_XTIME_DATASETS_BPM25 = "XTIME_DATASETS_BPM25"
"""Environment variable that points to a directory with BPM25 dataset."""

_BPM25_HOME_PAGE = "https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data"
"""Dataset home page."""

_BPM25_DATASET_FILE = "PRSA_data_2010.1.1-2014.12.31.csv"
"""File containing raw (unprocessed) BPM25 dataset that is located inside _XTIME_DATASETS_BPM25 directory."""


class BPM25Builder(DatasetBuilder):
    """BPM25: Beijing PM2.5 Data.

    This hourly data set contains the PM2.5 data of US Embassy in Beijing. 
    Meanwhile, meteorological data from Beijing Capital International Airport are also included.
    The dataâ€™s time period is between Jan 1st, 2010 to Dec 31st, 2014. Missing data are denoted as `NA`:
        https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
    """

    NAME = "beijing_pm_2_5"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset)
        self.encoder = TimeSeriesEncoderV1()

    def _check_pre_requisites(self) -> None:
        # Check raw dataset exists.
        if _XTIME_DATASETS_BPM25 not in os.environ:
            raise RuntimeError(
                f"No environment variable found `{_XTIME_DATASETS_BPM25}` that should point to a directory with "
                f"BPM25 (Beijing PM2.5 Data) dataset `{_BPM25_DATASET_FILE}` that can be downloaded from `{_BPM25_HOME_PAGE}`."
            )
        self._dataset_dir = Path(os.environ[_XTIME_DATASETS_BPM25]).absolute()
        if self._dataset_dir.is_file():
            self._dataset_dir = self._dataset_dir.parent
        if not (self._dataset_dir / _BPM25_DATASET_FILE).is_file():
            raise RuntimeError(
                f"BPM25 dataset location was identified as `{self._dataset_dir}`, but this is either not a directory "
                f"or dataset file (`{_BPM25_DATASET_FILE}`) not found in this location. Please, download `{_BPM25_DATASET_FILE}` of this "
                f"dataset from its home page `{_BPM25_HOME_PAGE}`."
            )

        # Check `tsfresh` library can be imported.
        try:
            import tsfresh.feature_extraction.feature_calculators as ts_features

        except ImportError:
            raise RuntimeError(
                f"The BPM25 dataset requires `tsfresh` library to compute ML features. If it has not been installed, "
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

        train_df = pd.read_csv(self._dataset_dir / (_BPM25_DATASET_FILE[0:-4] + "-default-train.csv"))
        test_df = pd.read_csv(self._dataset_dir / (_BPM25_DATASET_FILE[0:-4] + "-default-test.csv"))

        # These are the base feature names (will have prefixes such as `x_`, `y_` and `z_`). This check needs to be
        # consistent with feature generation in `_create_default_dataset` method.
        feature_names = self.encoder.features()
        features = [
            Feature(f"{axis}_{name}", FeatureType.CONTINUOUS) for axis, name in product(("x", "y", "z"), feature_names)
        ]

        # Check that data frames contains expected columns (3 is for three axes, 1 is for label).
        assert train_df.shape[1] == 3 * len(feature_names) + 1, "Train data frame contains wrong number of columns."
        assert test_df.shape[1] == 3 * len(feature_names) + 1, "Test data frame contains wrong number of columns."
        for feature in features:
            assert feature.name in train_df.columns, \
                f"Missing column `{feature}` in train dataframe (columns={list(train_df.columns)})."
            assert feature.name in test_df.columns, \
                f"Missing column `{feature}` in test dataframe (columns={list(train_df.columns)})."

        label: str = "activity"

        # Encode labels (that are strings here) into numerical representation (0, num_classes-1).
        label_encoder = LabelEncoder().fit(train_df[label])
        train_df[label] = label_encoder.transform(train_df[label])
        test_df[label] = label_encoder.transform(test_df[label])

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=BPM25Builder.NAME,
                version="default",
                task=ClassificationTask(TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=6),
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
        """Clean raw BPM25 dataset."""
        # Do not clean it again if it has already been cleaned.
        # Dataset provides a single file in `.csv` format with missing values
        # We will use the raw file directly
        _clean_dataset_file = (self._dataset_dir / _BPM25_DATASET_FILE).with_suffix(".csv")
        if _clean_dataset_file.is_file():
            return 
    

    def _create_default_dataset(self) -> None:
        """Create default train/test splits and save them to files.

        Input to this function is the clean dataset created by the `_clean_dataset` method of this class.
        """
        # Do not generate datasets if they have already been generated.
        default_train_dataset_file = self._dataset_dir / (_BPM25_DATASET_FILE[0:-4] + "-default-train.csv")
        default_test_dataset_file = self._dataset_dir / (_BPM25_DATASET_FILE[0:-4] + "-default-test.csv")
        if default_train_dataset_file.is_file() and default_test_dataset_file.is_file():
            return

        # Load clean dataset into a data frame (user_id,activity,timestamp,x,y,z)
        clean_dataset_file = (self._dataset_dir / _BPM25_DATASET_FILE).with_suffix(".csv")
        assert clean_dataset_file.is_file(), f"Clean dataset does not exist (this is internal error)."

        dtypes = {
            "month"  : "int64",
            "day  "  : "int64",
            "hour "  : "int64",
            "pm2.5"  : "float64",
            "DEWP "  : "int64",
            "TEMP "  : "float64",
            "PRES "  : "float64",
            "cbwd "  : "string",
            "Iws  "  : "float64",
            "Is   "  : "int64",
            "Ir   "  : "int64",
        }
        df: pd.DataFrame = pd.read_csv(clean_dataset_file, dtype=dtypes)
        print(f' --- Reading the {clean_dataset_file} file --- ')
        print(df.head())
        
        # Drop rows with missing(NaN) values
        df.dropna(axis=0, how="any", inplace=True)
        print(df.head())
        
        # Drop `No` column because we don't need it and check if there are 12 columns
        df = df.drop('No', axis=1)
        assert df.shape[1] == 12, f"Clean dataset expected to have 12 columns (shape={df.shape})."
        print(df.head())
        
        # Column `cbwd` - combined wind directions
        # Let's get all unique values of `cbwd`
        #for col in df:
        #    print(df['cbwd'].unique())
        # There should be only 4 unique values : ['SE' 'cv' 'NW' 'NE']
        
        
        ## We do not normalize accelerometer values (x, y, z) because trees are invariant to these transformations
        #
        ## Apply sliding window transformation.
        #window_size, stride = 200, 40
        #
        #train_windows = TimeSeries.slide(df_train[["x", "y", "z"]], None, window_size, stride)
        #train_labels = TimeSeries.slide(df_train.activity, TimeSeries.mode, window_size, stride)
        #test_windows = TimeSeries.slide(df_test[["x", "y", "z"]], None, window_size, stride)
        #test_labels = TimeSeries.slide(df_test.activity, TimeSeries.mode, window_size, stride)
        #
        #def _create_dataset(_windows: np.ndarray, _labels: np.ndarray, _file_path: Path) -> None:
        #    """Convert windows with raw accelerometer values into machine learning features and save to file.
        #    Args:
        #        _windows: Rank-3 tensor of shape [NumExamples, WindowSize, NumAxis].
        #        _labels: Array of labels, number of labels = NumExamples.
        #        _file_path: File name to write the generated dataset.
        #    """
        #    _dataset = pd.DataFrame(self.encoder.encode_many(_windows, prefixes=["x_", "y_", "z_"]))
        #    _dataset["activity"] = _labels.flatten()
        #    assert _dataset.shape[0] == _windows.shape[0], "error!"
        #    assert _dataset.shape[1] == 19 * 3 + 1, "error!"
        #    _dataset.to_csv(_file_path, index=False)
        #
        #_create_dataset(train_windows, train_labels, default_train_dataset_file)
        #_create_dataset(test_windows, test_labels, default_test_dataset_file)
