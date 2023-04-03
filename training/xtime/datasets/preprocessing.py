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
