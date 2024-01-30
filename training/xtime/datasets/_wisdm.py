import os
import typing as t
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from xtime.datasets import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

__all__ = ["WISDMBuilder"]


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

        if _XTIME_DATASETS_WISDM not in os.environ:
            raise RuntimeError(
                f"No environment variable found ({_XTIME_DATASETS_WISDM}) that should point to a directory with "
                f"WISDM (WIreless Sensor Data Mining) dataset v1.1 that can be downloaded from `{_WISDM_HOME_PAGE}`."
            )
        self._dataset_dir = Path(os.environ[_XTIME_DATASETS_WISDM]).absolute()
        if self._dataset_dir.is_file():
            self._dataset_dir = self._dataset_dir.parent
        if not (self._dataset_dir / _WISDM_DATASET_FILE).is_file():
            raise RuntimeError(
                f"WISDM dataset location was identified as `{self._dataset_dir}`, but this is either not a directory "
                f"or dataset file (`{_WISDM_DATASET_FILE}`) not found in this location. Please, download v1.1 of this "
                f"dataset from its home page `{_WISDM_HOME_PAGE}`."
            )

        try:
            import tsfresh.feature_extraction.feature_calculators as ts_features

            self._ts_features = ts_features
        except ImportError:
            raise RuntimeError(
                f"The WISDM dataset requires `tsfresh` library to compute ML features. If it has not been installed, "
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

        train_df = pd.read_csv(self._dataset_dir / (_WISDM_DATASET_FILE[0:-4] + "-default-train.csv"))
        test_df = pd.read_csv(self._dataset_dir / (_WISDM_DATASET_FILE[0:-4] + "-default-test.csv"))

        # These are the base feature names (will have prefixes such as `x_`, `y_` and `z_`). This check needs to be
        # consistent with feature generation in `_create_default_dataset` method.
        base_names = [
            "abs_energy",
            "absolute_sum_of_changes",
            "count_above_mean",
            "kurtosis",
            "longest_strike_above_mean",
            "longest_strike_below_mean",
            "maximum",
            "mean",
            "mean_abs_change",
            "mean_change",
            "median",
            "minimum",
            "number_crossing_0",
            "quantile_25",
            "quantile_75",
            "rms",
            "skewness",
            "sum_values",
            "variance",
        ]

        features = []
        for axis in ("x", "y", "z"):
            for base_name in base_names:
                features.append(Feature(f"{axis}_{base_name}", FeatureType.CONTINUOUS))

        # Check that data frames contains expected columns (3 is for three axes, 1 is for label).
        assert train_df.shape[1] == 3 * len(base_names) + 1, "Train data frame contains unexpected number of columns."
        assert test_df.shape[1] == 3 * len(base_names) + 1, "Test data frame contains unexpected number of columns."
        for feature in features:
            assert feature.name in train_df.columns, f"Missing column `{feature}` in train dataframe."
            assert feature.name in test_df.columns, f"Missing column `{feature}` in test dataframe."

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
                        print(f"Empty line (line_no={idx + 1})")
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
                            print(
                                "Line contains an instance with wrong number of columns:",
                                "line_no:",
                                idx + 1,
                                "line:",
                                line,
                                "instance:",
                                instance,
                            )
                        elif activity not in class_names:
                            print("Invalid class name:", "line_no:", idx + 1, "class_name:", activity)
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
        train_windows, train_labels = _slide(df_train[["x", "y", "z"]], df_train.activity, window_size, stride)
        test_windows, test_labels = _slide(df_test[["x", "y", "z"]], df_test.activity, window_size, stride)

        def _features(_ts: np.ndarray, _name: str) -> t.Dict:
            """Compute features for the given uni-variate time series segment.
            Args:
                _ts: Uni-variate time series that `tsfresh` can process.
                _name: Axis name (one of `x`, `y` or `z`).
            Returns:
                 Dictionary mapping feature names for feature values.
            """
            features = {
                "abs_energy": self._ts_features.abs_energy(_ts),  # float
                "absolute_sum_of_changes": self._ts_features.absolute_sum_of_changes(_ts),  # float
                "count_above_mean": self._ts_features.count_above_mean(_ts),  # float
                "kurtosis": self._ts_features.kurtosis(_ts),  # float
                "longest_strike_above_mean": self._ts_features.longest_strike_above_mean(_ts),  # float
                "longest_strike_below_mean": self._ts_features.longest_strike_below_mean(_ts),  # float
                "maximum": self._ts_features.maximum(_ts),  # float
                "mean": self._ts_features.mean(_ts),  # float
                "mean_abs_change": self._ts_features.mean_abs_change(_ts),  # float
                "mean_change": self._ts_features.mean_change(_ts),  # float
                "median": self._ts_features.median(_ts),  # float
                "minimum": self._ts_features.minimum(_ts),  # float
                "number_crossing_0": self._ts_features.number_crossing_m(_ts, 0),  # float
                "quantile_25": self._ts_features.quantile(_ts, 0.25),  # float
                "quantile_75": self._ts_features.quantile(_ts, 0.75),  # float
                "rms": self._ts_features.root_mean_square(_ts),  # float
                "skewness": self._ts_features.skewness(_ts),  # float
                "sum_values": self._ts_features.sum_values(_ts),  # float
                "variance": self._ts_features.variance(_ts),  # float
            }
            return {f"{_name}_{k}": v for k, v in features.items()}

        def _windows_to_features(_windows: np.ndarray) -> t.List[t.Dict]:
            """Convert list fo raw window data into list of corresponding features.
            Args:
                _windows: Rank-3 tensor of shape [NumExamples, WindowSize, NumAxis].
            Returns:
                List of _windows.shape[0] size with each element being a dictionary mapping feature names to feature
                    values for rank-2 [WindowSize, NumAxis] tensors.
            """
            assert _windows.ndim == 3 and _windows.shape[1] == window_size and _windows.shape[2] == 3, "error!"
            _features_list: t.List[t.Dict] = []
            for i in range(_windows.shape[0]):
                _features_list.append(
                    {
                        **_features(_windows[i, :, 0], "x"),
                        **_features(_windows[i, :, 1], "y"),
                        **_features(_windows[i, :, 2], "z"),
                    }
                )
            return _features_list

        def _create_dataset(_windows: np.ndarray, _labels: np.ndarray, _file_path: Path) -> None:
            """Convert windows with raw accelerometer values into machine learning features and save to file.
            Args:
                _windows: Rank-3 tensor of shape [NumExamples, WindowSize, NumAxis].
                _labels: Array of labels, number of labels = NumExamples.
                _file_path: File name to write the generated dataset.
            """
            _dataset = pd.DataFrame(_windows_to_features(_windows))
            _dataset["activity"] = _labels.flatten()
            assert _dataset.shape[0] == _windows.shape[0], "error!"
            assert _dataset.shape[1] == 19 * 3 + 1, "error!"
            _dataset.to_csv(_file_path, index=False)

        _create_dataset(train_windows, train_labels, default_train_dataset_file)
        _create_dataset(test_windows, test_labels, default_test_dataset_file)


def _slide(
    raw_vals: pd.DataFrame, y: pd.Series, window_size: int = 1, stride: int = 1
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Apply sliding window transformation and return windows with raw accelerometer values and corresponding labels.
    Args:
        raw_vals: Data frame with raw accelerometer values (3 columns - `x`, `y` and `z`).
        y: Labels for each row in `raw_vals`.
        window_size: Size of the sliding window.
        stride: Window step size.
    Returns:
        A tuple containing two numpy arrays (windows, labels). The window tensor is a rank-3 tensor of shape
        [NumWindows, WindowSize, NumAxis] where NumAxis is 3.
    """
    assert raw_vals.shape[1] == 3, "Expecting accelerometer values for 3 axes."
    windows, labels = [], []
    for i in range(0, len(raw_vals) - window_size, stride):
        windows.append(
            # Take `window_size` rows (time steps)
            raw_vals.iloc[i : (i + window_size)].values
        )
        labels.append(
            # Identify the most common label in this window (1 most common element returning list of (element, count))
            Counter(y.iloc[i : i + window_size]).most_common(1)[0][0]
        )
    _windows, _labels = np.array(windows), np.array(labels).reshape(-1, 1)
    assert _windows.ndim == 3, "Invalid train windows shape"
    assert _windows.shape[1] == window_size, "Invalid train windows shape"
    assert _windows.shape[2] == 3, "Invalid train windows shape"
    return _windows, _labels
