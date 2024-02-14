import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # archit
from sklearn.preprocessing import LabelEncoder

from xtime.datasets import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from xtime.datasets.preprocessing import TimeSeriesEncoderV1
from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

__all__ = ["OLD1HRBuilder"]

logger = logging.getLogger(__name__)


_XTIME_DATASETS_OLD1HR = "XTIME_DATASETS_OLD1HR"
"""Environment variable that points to a directory with Ozone Level Detection (OLD) dataset."""

_OLD1HR_HOME_PAGE = "https://archive.ics.uci.edu/dataset/172/ozone+level+detection"
"""Dataset home page."""

_OLD1HR_DATASET_FILE = "onehr.data"
"""File containing raw (unprocessed) OLD1HR dataset that is located inside _XTIME_DATASETS_OLD1HR directory."""


class OLD1HRBuilder(DatasetBuilder):
    """OLD1HR: Ozone Level Detection.

    Two ground ozone level data sets are included in this collection.
    One is the eight hour peak set (eighthr.data), the other is the one hour peak set (onehr.data).
    Those data were collected from 1998 to 2004 at the Houston, Galveston and Brazoria area.
    For a list of attributes, please refer to those two .names files.
        https://archive.ics.uci.edu/dataset/172/ozone+level+detection
    """

    NAME = "ozone_level_detection_1hr"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset)
        self.encoder = TimeSeriesEncoderV1()

    def _check_pre_requisites(self) -> None:
        # Check raw dataset exists.
        if _XTIME_DATASETS_OLD1HR not in os.environ:
            raise RuntimeError(
                f"No environment variable found (`{_XTIME_DATASETS_OLD1HR}`) that should point to a directory with "
                f"OLD1HR (Ozone Level Detection) dataset (`{_OLD1HR_DATASET_FILE}`) that can be downloaded from `{_OLD1HR_HOME_PAGE}`."
            )
        self._dataset_dir = Path(os.environ[_XTIME_DATASETS_OLD1HR]).absolute()
        if self._dataset_dir.is_file():
            self._dataset_dir = self._dataset_dir.parent
        if not (self._dataset_dir / _OLD1HR_DATASET_FILE).is_file():
            raise RuntimeError(
                f"OLD1HR dataset location was identified as `{self._dataset_dir}`, but this is either not a directory "
                f"or dataset file (`{_OLD1HR_DATASET_FILE}`) not found in this location. Please, download (`{_OLD1HR_DATASET_FILE}`) of this "
                f"dataset from its home page `{_OLD1HR_HOME_PAGE}`."
            )

        # Check `tsfresh` library can be imported.
        try:
            import tsfresh.feature_extraction.feature_calculators as ts_features

        except ImportError:
            raise RuntimeError(
                f"The OLD1HR dataset requires `tsfresh` library to compute ML features. If it has not been installed, "
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

        train_df = pd.read_csv(self._dataset_dir / (_OLD1HR_DATASET_FILE + "-default-train.csv"))
        test_df = pd.read_csv(self._dataset_dir / (_OLD1HR_DATASET_FILE + "-default-test.csv"))

        # feature_names = self.encoder.features()
        # All features in this dataset are continuous (float64) except first column (Date) which we are dropping
        features = [
            Feature(col, FeatureType.CONTINUOUS, cardinality=int(train_df[col].nunique())) for col in train_df.columns
        ]

        label: str = "label"

        # Encode labels (that are strings here) into numerical representation (0, num_classes-1).
        label_encoder = LabelEncoder().fit(train_df[label])
        train_df[label] = label_encoder.transform(train_df[label])
        test_df[label] = label_encoder.transform(test_df[label])

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=OLD1HRBuilder.NAME,
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
        """Clean raw OLD1HR dataset."""
        # Do not clean it again if it has already been cleaned.
        _clean_dataset_file = (self._dataset_dir / _OLD1HR_DATASET_FILE).with_suffix(".csv")
        if _clean_dataset_file.is_file():
            return

        with open(self._dataset_dir / _OLD1HR_DATASET_FILE, "rt") as input_stream:
            with open(_clean_dataset_file, "wt") as output_stream:
                # output_stream.write("user_id,activity,timestamp,x,y,z\n")
                for idx, line in enumerate(input_stream):
                    line = line.strip("\t\n")
                    if not line:
                        # Skip all empty lines.
                        logger.debug("Empty line (line_no=%s).", idx + 1)
                        continue
                    for instance in line.split("\t"):
                        # Raw dataset contains `\t` at the end of each line.
                        instance = instance.strip(" ,")
                        output_stream.write(instance + "\n")

    def _create_default_dataset(self) -> None:
        """Create default train/test splits and save them to files.

        Input to this function is the clean dataset created by the `_clean_dataset` method of this class.
        """
        # Do not generate datasets if they have already been generated.
        default_train_dataset_file = self._dataset_dir / (_OLD1HR_DATASET_FILE + "-default-train.csv")
        default_test_dataset_file = self._dataset_dir / (_OLD1HR_DATASET_FILE + "-default-test.csv")
        if default_train_dataset_file.is_file() and default_test_dataset_file.is_file():
            return

        # Load clean dataset into a data frame (Date, 72 continuous features, labels (two classes 1: ozone day, 0: normal day))
        clean_dataset_file = (self._dataset_dir / _OLD1HR_DATASET_FILE).with_suffix(".csv")
        assert clean_dataset_file.is_file(), f"Clean dataset does not exist (this is internal error)."

        df: pd.DataFrame = pd.read_csv(clean_dataset_file, delimiter=",", header=None)

        # dataset doesn't have feature_names but the information is provided
        feature_names_df = pd.read_csv(self._dataset_dir / "feature_names.csv", header=None)

        df.columns = feature_names_df[0].tolist()

        df_with_feature_names = self._dataset_dir / (_OLD1HR_DATASET_FILE + "_with_feature_names.csv")
        df.to_csv(df_with_feature_names, index=False)

        # Following some of the examples from: https://www.kaggle.com/datasets/prashant111/ozone-level-detection/code
        # and https://github.com/aaakashkumar/Ozone-Level-Detection/blob/master/DSDA_Project_%E2%80%94_Ozone_Level_Detection.ipynb
        df = df.drop(df.columns[0], axis=1)

        df.replace(to_replace="?", value=np.nan, inplace=True)
        df.dropna(axis=0, how="any", inplace=True)
        assert df.shape[1] == 73, f"Clean dataset expected to have 73 columns (shape={df.shape})."

        # Split into train/test subsets
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

        df_train.to_csv(default_train_dataset_file, index=False)
        df_test.to_csv(default_test_dataset_file, index=False)
