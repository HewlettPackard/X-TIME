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
import abc
import typing as t
from collections import Counter

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

from xtime.ml import Feature, FeatureType


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols: t.Union[t.List[str], str]) -> None:
        if not isinstance(cols, list):
            cols = [cols]
        self.cols = cols

    def fit(self, *_args, **_kwargs) -> "DropColumns":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x.drop(self.cols, axis=1, inplace=True)
        return x


class ChangeColumnsType(BaseEstimator, TransformerMixin):
    def __init__(self, cols: t.Union[t.List[str], str], dtype: t.Any) -> None:
        if not isinstance(cols, list):
            cols = [cols]
        self.cols = cols
        self.dtype = dtype

    def fit(self, *_args, **_kwargs) -> "ChangeColumnsType":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        for col in self.cols:
            x[col] = x[col].astype(self.dtype)
        return x


class ChangeColumnsTypeToCategory(BaseEstimator, TransformerMixin):
    def __init__(self, features: t.List[Feature]) -> None:
        self.features = features

    def fit(self, *_args, **_kwargs) -> "ChangeColumnsTypeToCategory":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        for feature in self.features:
            if feature.type.categorical():
                x[feature.name] = x[feature.name].astype(
                    CategoricalDtype(ordered=(feature.type == FeatureType.ORDINAL))
                )
        return x


class CheckColumnsOrder(BaseEstimator, TransformerMixin):
    def __init__(self, cols: t.Union[t.List[str], str], label: t.Optional[str] = None) -> None:
        if not isinstance(cols, list):
            cols = [cols]
        self.cols = cols
        self.label = label

    def fit(self, *_args, **_kwargs) -> "CheckColumnsOrder":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        actual_cols = x.columns.to_list()
        if self.label:
            actual_cols.remove(self.label)
        if len(actual_cols) != len(self.cols):
            raise ValueError(
                f"CheckColumnsOrder failed: len(x.columns) != len(self.cols) ({len(actual_cols)} != {len(self.cols)})"
            )
        for idx, (actual, expected) in enumerate(zip(actual_cols, self.cols)):
            if actual != expected:
                raise ValueError(
                    f"CheckColumnsOrder failed: actual != expected ({actual} != {expected}) at index {idx}."
                )
        return x


class EncodeCategoricalColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols: t.Union[t.List[str], str]) -> None:
        if not isinstance(cols, list):
            cols = [cols]
        self.cols = cols
        self.encoders: t.List[OrdinalEncoder] = []

    def fit(self, x: pd.DataFrame, *_args, **_kwargs) -> "EncodeCategoricalColumns":
        self.encoders = []
        for col in self.cols:
            self.encoders.append(OrdinalEncoder(dtype=int).fit(x[col].values.reshape(-1, 1)))
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        for col, encoder in zip(self.cols, self.encoders):
            x[col] = encoder.transform(x[col].values.reshape(-1, 1))
        return x


class TimeSeries:
    """Timeseries related methods.

    Time series are usually stored as rank-2 or rank-3 tensors, and layout of these tensors is determined by what
    data is stored there - time series or time series segments.
        - Time series are stored as rank-2 tensors where individual times series are stored column-wise: each column
          contains one time series, while each row contains values for multiple time series for the same time step.
        - Segments of time series are stores as rank-2 or rank-3 tensors, though rank-2 tensors are reshaped as
          rank-3 tensors internally:
           - rank-2 tensors have the following shape: [NumSegments, SegmentLength].
           - rank-3 tensors have the following shape: [NumSegments, SegmentLength, NumTimeSeries]. For univariate time
             series the last dimension is always 1.
    """

    @staticmethod
    def mode(timeseries: t.Any) -> t.Any:
        """Identify the most common element in this timeseries segment.
        Args:
            timeseries: Time series segment.
        Return:
            The most common element.
        """
        if timeseries.ndim == 2 and (timeseries.shape[0] == 1 or timeseries.shape[1] == 1):
            timeseries = timeseries.flatten()
        assert timeseries.ndim == 1, f"Invalid rank or shape (shape = {timeseries.shape})"
        return Counter(timeseries).most_common(1)[0][0]

    @staticmethod
    def slide(
        timeseries: t.Union[pd.DataFrame, pd.Series],
        transform: t.Optional[t.Callable] = None,
        window_size: int = 1,
        stride: int = 1,
    ) -> np.ndarray:
        """Apply sliding window transformation to input uni-variate or multi-variate time series.
        Args:
            timeseries: Data frame with one or multiple time series. One column == one time series.
            transform: Transformation applied to every time series segment. It accepts rank-2
                (WindowSize, NumTimeseries) numpy array and returns transformed version (e.g., mean).
            window_size: Size of the sliding window.
            stride: Window step size.
        Returns:
            Rank-3 (NumSegments, WindowSize, NumTimeseries) numpy array.
        """
        has_transform = transform is not None
        if not has_transform:
            transform = transform if transform is not None else (lambda segment: segment)
        length: int = timeseries.shape[0]
        if window_size > length:
            raise ValueError(f"Window size ({window_size}) is bigger than timeseries length ({length}).")

        cursor, windows = 0, []
        while True:
            if cursor + window_size > length:
                break
            windows.append(transform(timeseries.iloc[cursor : (cursor + window_size)].values))
            cursor += stride

        _windows = np.array(windows)
        if not has_transform:
            # With transform (such as computing segment mode) we can't conduct these tests.
            assert _windows.ndim == 3, f"Invalid train windows shape (shape={_windows.shape})."
            assert _windows.shape[1] == window_size, \
                f"Invalid train windows shape (shape={_windows.shape}, window_size={window_size})."
        return _windows


class TimeSeriesEncoder(abc.ABC):
    """Encoding time series segments for ML model models."""

    @abc.abstractmethod
    def features(self) -> t.List[str]:
        """Return list of feature names in this encoder."""
        raise NotImplementedError

    @abc.abstractmethod
    def encode(
        self, segment: np.ndarray, prefix: t.Optional[str] = None, suffix: t.Optional[str] = None
    ) -> t.Dict[str, t.Union[float, int]]:
        """Encode a time series segment using a pre-defined set of features from time domain.

        Describe this is per segment per axis

        Args:
            segment: Time series segment is a rank-1 or rank-2 numpy array. In the latter case, its shape must be
                (1, Length).
            prefix: Optional prefix for feature names.
            suffix: Optional suffix for feature names.

        Returns:
            Dictionary mapping feature names to feature values.
        """
        raise NotImplementedError

    def encode_many(self, segments: np.ndarray, prefixes: t.Optional[t.List[str]] = None) -> t.List[t.Dict]:
        """Encode multiple segments.

        Args:
            segments: Rank-2 or rank-3 tensor:
                - rank-2 tensor: matrix containing segments for uni-variate time series, matrix shape is (NumSegments,
                  SegmentLength).
                - rank-3 tensor: tensor containing segments for multi-variate time series, tensor shape is (NumSegments,
                  SegmentLength, NumTimeSeries).
            prefixes: Optional prefixes to add to feature names. Prefixes are mandatory for multi-variate time series.
        Returns:
            List of dictionaries containing segment features.
        """
        if segments.ndim == 2:
            segments = segments.reshape(segments.shape[0], segments.shape[1], 1)
        assert segments.ndim == 3, (
            "Segments must be a rank-3 (NumSegments, SegmentSize, NumTimeSeries) "
            f"numpy array (shape = {segments.shape})."
        )

        if segments.shape[2] > 1 and (prefixes is None or segments.shape[2] != len(prefixes)):
            assert False, f"When number of axes > 1, the prefixes list must present (shape = {segments.shape})."

        if prefixes is None:
            assert segments.shape[2] == 1, f"Internal assert failed (shape = {segments.shape})."
            prefixes = [None]

        feature_list: t.List[t.Dict] = []
        for i in range(segments.shape[0]):
            features = {}
            for j in range(segments.shape[2]):
                features.update(self.encode(segments[i, :, j], prefix=prefixes[j]))
            feature_list.append(features)
        return feature_list

    @staticmethod
    def normalize_segment(segment: np.ndarray) -> np.ndarray:
        """Normalize a time series segment.

        Normalization means ensuring the type of time series segment and its shape is the one expected by this library.

        Args:
            segment: Time series segment, usually obtained by applying sliding window transformation to time series.
                Valid times series are numpy arrays that are either rank-1 tensors (ndim == 1), or rank-2 tensor
                with shape[0] == 1. This implies that matrices representing machine learning datasets must contain
                time series segments row-wise (one row == one segment), in other words, with shape = (NumSegments,
                SegmentLength).
        Returns:
            Original input as is or its flattened version (ndim is always 1).
        """
        assert isinstance(
            segment, np.ndarray
        ), f"Unsupported time series format (type = {type(segment)}). Expected numpy array."
        if segment.ndim == 2:
            assert segment.shape[0] == 1, (
                f"Unsupported times series dimension (shape = {segment.shape}). "
                "When segment is a rank-2 tensor, its shape must be (1, Length)."
            )
            segment = segment.flatten()
        assert segment.ndim == 1, (
            f"Unsupported time series dimension (shape = {segment.shape}). "
            "It should be either a rank-1 tensor or rank-2 tensor with (1, Length) shape."
        )
        return segment


class TimeSeriesEncoderV1(TimeSeriesEncoder):
    def __init__(self) -> None:
        import tsfresh.feature_extraction.feature_calculators as ts_features

        self._feature_calculators: t.Dict[str, t.Callable[[np.ndarray], t.Union[float, int]]] = {
            "abs_energy": lambda ts: float(ts_features.abs_energy(ts)),
            "absolute_sum_of_changes": lambda ts: float(ts_features.absolute_sum_of_changes(ts)),
            "count_above_mean": lambda ts: int(ts_features.count_above_mean(ts)),
            "kurtosis": lambda ts: float(ts_features.kurtosis(ts)),
            "longest_strike_above_mean": lambda ts: int(ts_features.longest_strike_above_mean(ts)),
            "longest_strike_below_mean": lambda ts: int(ts_features.longest_strike_below_mean(ts)),
            "maximum": lambda ts: float(ts_features.maximum(ts)),
            "mean": lambda ts: float(ts_features.mean(ts)),
            "mean_abs_change": lambda ts: float(ts_features.mean_abs_change(ts)),
            "mean_change": lambda ts: float(ts_features.mean_change(ts)),
            "median": lambda ts: float(ts_features.median(ts)),
            "minimum": lambda ts: float(ts_features.minimum(ts)),
            "number_crossing_0": lambda ts: int(ts_features.number_crossing_m(ts, m=0)),
            "quantile_25": lambda ts: float(ts_features.quantile(ts, q=0.25)),
            "quantile_75": lambda ts: float(ts_features.quantile(ts, q=0.75)),
            "rms": lambda ts: float(ts_features.root_mean_square(ts)),
            "skewness": lambda ts: float(ts_features.skewness(ts)),
            "sum_values": lambda ts: float(ts_features.sum_values(ts)),
            "variance": lambda ts: float(ts_features.variance(ts)),
        }

    def features(self) -> t.List[str]:
        return list(self._feature_calculators.keys())

    def encode(
        self, segment: np.ndarray, prefix: t.Optional[str] = None, suffix: t.Optional[str] = None
    ) -> t.Dict[str, t.Union[float, int]]:
        segment = TimeSeriesEncoderV1.normalize_segment(segment)
        features = {name: calculate(segment) for name, calculate in self._feature_calculators.items()}
        if prefix or suffix:
            prefix, suffix = prefix or "", suffix or ""
            features = {f"{prefix}{k}{suffix}": v for k, v in features.items()}

        return features
