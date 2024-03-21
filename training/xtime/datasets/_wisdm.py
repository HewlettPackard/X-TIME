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
from sklearn.preprocessing import LabelEncoder

from xtime.datasets import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from xtime.datasets.dataset import DatasetPrerequisites
from xtime.datasets.preprocessing import TimeSeries, TimeSeriesEncoderV1
from xtime.errors import DatasetError
from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

__all__ = ["WISDMBuilder"]

logger = logging.getLogger(__name__)


_XTIME_DATASETS_WISDM = "XTIME_DATASETS_WISDM"
"""Environment variable that points to a directory with WISDM dataset."""

_WISDM_HOME_PAGE = "https://www.cis.fordham.edu/wisdm/dataset.php"
"""Dataset home page."""

_WISDM_DATASET_FILE = "WISDM_ar_v1.1_raw.txt"
"""File containing raw (unprocessed) WISDM dataset that is located inside _XTIME_DATASETS_WISDM directory."""


class WISDMBuilder(DatasetBuilder):
    """WISDM: WIreless Sensor Data Mining.

    Activity classification using 3-axis on-body accelerometer. This implementation is partially based on
    Towards Data Science blog by Venelin Valkov (Time Series Classification for Human Activity Recognition with LSTMs
    using TensorFlow 2 and Keras):
        https://towardsdatascience.com/time-series-classification-for-human-activity-recognition-with-lstms-using-tensorflow-2-and-keras-b816431afdff
    """

    NAME = "wisdm"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset)
        self.encoder = TimeSeriesEncoderV1()

    def _check_pre_requisites(self) -> None:
        # Check raw dataset exists.
        if _XTIME_DATASETS_WISDM not in os.environ:
            raise DatasetError.missing_prerequisites(
                f"No environment variable found ({_XTIME_DATASETS_WISDM}) that should point to a directory with "
                f"WISDM (WIreless Sensor Data Mining) dataset v1.1 that can be downloaded from `{_WISDM_HOME_PAGE}`."
            )
        self._dataset_dir = Path(os.environ[_XTIME_DATASETS_WISDM]).absolute()
        if self._dataset_dir.is_file():
            self._dataset_dir = self._dataset_dir.parent
        if not (self._dataset_dir / _WISDM_DATASET_FILE).is_file():
            raise DatasetError.missing_prerequisites(
                f"WISDM dataset location was identified as `{self._dataset_dir}`, but this is either not a directory "
                f"or dataset file (`{_WISDM_DATASET_FILE}`) not found in this location. Please, download v1.1 of this "
                f"dataset from its home page `{_WISDM_HOME_PAGE}`."
            )

        # Check `tsfresh` library can be imported.
        DatasetPrerequisites.check_tsfresh(self.NAME)

    def _build_default_dataset(self, **kwargs) -> Dataset:
        if kwargs:
            raise ValueError(f"{self.__class__.__name__}: `default` dataset does not accept arguments.")
        self._clean_dataset()
        self._create_default_dataset()

        train_df = pd.read_csv(self._dataset_dir / (_WISDM_DATASET_FILE[0:-4] + "-default-train.csv"))
        test_df = pd.read_csv(self._dataset_dir / (_WISDM_DATASET_FILE[0:-4] + "-default-test.csv"))

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
            assert (
                feature.name in train_df.columns
            ), f"Missing column `{feature}` in train dataframe (columns={list(train_df.columns)})."
            assert (
                feature.name in test_df.columns
            ), f"Missing column `{feature}` in test dataframe (columns={list(train_df.columns)})."

        label: str = "activity"

        # Encode labels (that are strings here) into numerical representation (0, num_classes-1).
        label_encoder = LabelEncoder().fit(train_df[label])
        train_df[label] = label_encoder.transform(train_df[label])
        test_df[label] = label_encoder.transform(test_df[label])

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=WISDMBuilder.NAME,
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
        """Clean raw WISDM dataset."""
        # Do not clean it again if it has already been cleaned.
        _clean_dataset_file = (self._dataset_dir / _WISDM_DATASET_FILE).with_suffix(".csv")
        if _clean_dataset_file.is_file():
            return

        # These are class names expected to be present. No other class names should present.
        class_names = {"Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"}

        with open(self._dataset_dir / _WISDM_DATASET_FILE, "rt") as input_stream:
            with open(_clean_dataset_file, "wt") as output_stream:
                output_stream.write("user_id,activity,timestamp,x,y,z\n")
                for idx, line in enumerate(input_stream):
                    line = line.strip(" ;\n")
                    if not line:
                        # Skip all empty lines.
                        logger.debug("Empty line (line_no=%s).", idx + 1)
                        continue
                    for instance in line.split(";"):
                        # Raw dataset contains `;` at the end of each line.
                        instance = instance.strip(" ,")
                        if not instance:
                            continue
                        columns = instance.split(",")
                        activity = columns[1].strip()
                        if len(columns) != 6:
                            # Some lines contains missing values (one of x,y,z accelerometer value).
                            logger.debug(
                                "Line contains an instance with wrong number of columns: line_no: %d, "
                                "line: %s, instance: %s.",
                                idx + 1,
                                line,
                                instance,
                            )
                        elif activity not in class_names:
                            logger.debug("Invalid class name: line_no: %d, class_name: %s.", idx + 1, activity)
                        else:
                            output_stream.write(instance + "\n")

    def _create_default_dataset(self) -> None:
        """Create default train/test splits and save them to files.

        Input to this function is the clean dataset created by the `_clean_dataset` method of this class.
        """
        # Do not generate datasets if they have already been generated.
        default_train_dataset_file = self._dataset_dir / (_WISDM_DATASET_FILE[0:-4] + "-default-train.csv")
        default_test_dataset_file = self._dataset_dir / (_WISDM_DATASET_FILE[0:-4] + "-default-test.csv")
        if default_train_dataset_file.is_file() and default_test_dataset_file.is_file():
            return

        # Load clean dataset into a data frame (user_id,activity,timestamp,x,y,z)
        clean_dataset_file = (self._dataset_dir / _WISDM_DATASET_FILE).with_suffix(".csv")
        assert clean_dataset_file.is_file(), f"Clean dataset does not exist (this is internal error)."
        dtypes = {
            "user_id": "int64",
            "activity": "string",
            "timestamp": "int64",
            "x": "float64",
            "y": "float64",
            "z": "float64",
        }
        df: pd.DataFrame = pd.read_csv(clean_dataset_file, dtype=dtypes)
        df.dropna(axis=0, how="any", inplace=True)
        assert df.shape[1] == 6, f"Clean dataset expected to have 6 columns (shape={df.shape})."
        for col in dtypes.keys():
            assert col in df.columns, f"Clean dataset does not provdie `{col}` column."

        # Split into train/test subsets
        df_train = df[df["user_id"] <= 30]
        df_test = df[df["user_id"] > 30]

        # We do not normalize accelerometer values (x, y, z) because trees are invariant to these transformations

        # Apply sliding window transformation.
        window_size, stride = 200, 40

        train_windows = TimeSeries.slide(df_train[["x", "y", "z"]], None, window_size, stride)
        train_labels = TimeSeries.slide(df_train.activity, TimeSeries.mode, window_size, stride)
        test_windows = TimeSeries.slide(df_test[["x", "y", "z"]], None, window_size, stride)
        test_labels = TimeSeries.slide(df_test.activity, TimeSeries.mode, window_size, stride)

        def _create_dataset(_windows: np.ndarray, _labels: np.ndarray, _file_path: Path) -> None:
            """Convert windows with raw accelerometer values into machine learning features and save to file.
            Args:
                _windows: Rank-3 tensor of shape [NumExamples, WindowSize, NumAxis].
                _labels: Array of labels, number of labels = NumExamples.
                _file_path: File name to write the generated dataset.
            """
            _dataset = pd.DataFrame(self.encoder.encode_many(_windows, prefixes=["x_", "y_", "z_"]))
            _dataset["activity"] = _labels.flatten()
            assert _dataset.shape[0] == _windows.shape[0], "error!"
            assert _dataset.shape[1] == 19 * 3 + 1, "error!"
            _dataset.to_csv(_file_path, index=False)

        _create_dataset(train_windows, train_labels, default_train_dataset_file)
        _create_dataset(test_windows, test_labels, default_test_dataset_file)
