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
import typing as t
from functools import partial
from unittest import TestCase

import numpy as np
import pandas as pd

from xtime.datasets.preprocessing import TimeSeries, TimeSeriesEncoder, TimeSeriesEncoderV1


class TestTimeSeries(TestCase):
    def test_mode(self) -> None:
        classes = np.asarray([0, 1, 2, 3, 3, 4, 3, 2, 1, 1, 6, 7, 1, 1, 4, 4, 1])
        self.assertEqual(1, TimeSeries.mode(classes.flatten()))
        self.assertEqual(1, TimeSeries.mode(classes.reshape(-1, 1)))
        self.assertEqual(1, TimeSeries.mode(classes.reshape(1, -1)))

    def test_slide_no_transform(self) -> None:
        #
        length: int = 10
        df = pd.DataFrame({"a": np.random.randn(length), "b": np.random.randn(length)})

        self.assertRaises(ValueError, partial(self._slide, df, window_size=length + 1, stride=1))

        windows = self._slide(df, window_size=1, stride=1)
        self.assertEqual(length, windows.shape[0])

        windows = self._slide(df, window_size=length, stride=1)
        self.assertEqual(1, windows.shape[0])

        windows = self._slide(df, window_size=2, stride=2)
        self.assertEqual(5, windows.shape[0])

        windows = self._slide(df, window_size=3, stride=2)
        self.assertEqual(4, windows.shape[0])

    def _slide(self, df: pd.DataFrame, window_size: int, stride: int = 1) -> np.ndarray:
        windows = TimeSeries.slide(df, window_size=window_size, stride=stride)
        self.assertIsInstance(windows, np.ndarray)
        self.assertEqual(3, windows.ndim)
        self.assertEqual(window_size, windows.shape[1])
        self.assertEqual(df.shape[1], windows.shape[2])
        return windows


class TestTimeSeriesEncoder(TestCase):
    def _test(self, segment: np.ndarray) -> None:
        segment = TimeSeriesEncoder.normalize_segment(segment)
        self.assertIsInstance(segment, np.ndarray)
        self.assertEqual(1, segment.ndim)
        self.assertEqual(6, len(segment))

    def test_normalize_segment(self) -> None:
        arr = np.asarray([0, 1, 2, 3, 4, 6])

        self._test(arr)
        self._test(arr.reshape(1, -1))
        self.assertRaises(AssertionError, partial(self._test, arr.reshape(-1, 1)))
        self.assertRaises(AssertionError, partial(self._test, arr.reshape(2, 3)))


class TestTimeSeriesEncoderV1(TestCase):
    NUM_FEATURES = 19

    def test_features(self) -> None:
        features: t.List[str] = TimeSeriesEncoderV1().features()
        self.assertIsInstance(features, list)
        self.assertEqual(TestTimeSeriesEncoderV1.NUM_FEATURES, len(features))
        for feature in features:
            self.assertIsInstance(feature, str)

    def _check_features(self, features: t.Dict[str, t.Union[float, int]], num_ts: int = 1) -> None:
        self.assertIsInstance(features, dict)
        self.assertEqual(TestTimeSeriesEncoderV1.NUM_FEATURES * num_ts, len(features))
        for name, value in features.items():
            self.assertIsInstance(name, str)
            self.assertTrue(name.strip() == name)
            self.assertTrue(len(name) > 0)
            self.assertIsInstance(value, (float, int), f"actual type is `{type(value)}`.")

    def test_encode(self) -> None:
        encoder = TimeSeriesEncoderV1()

        segment: np.ndarray = np.random.randn(100)
        self._check_features(encoder.encode(segment))
        self._check_features(encoder.encode(segment.reshape(1, -1)))

    def test_encode_many(self) -> None:
        encoder = TimeSeriesEncoderV1()

        feature_list: t.List[t.Dict] = encoder.encode_many(np.random.randn(10, 100))
        self.assertIsInstance(feature_list, list)
        for features in feature_list:
            self._check_features(features)

        feature_list: t.List[t.Dict] = encoder.encode_many(np.random.randn(10, 100, 1))
        self.assertIsInstance(feature_list, list)
        for features in feature_list:
            self._check_features(features)

        feature_list: t.List[t.Dict] = encoder.encode_many(np.random.randn(10, 100, 2), prefixes=["x_", "y_"])
        self.assertIsInstance(feature_list, list)
        for features in feature_list:
            self._check_features(features, num_ts=2)
            for name in features.keys():
                self.assertTrue(name.startswith(("x_", "y_")))
