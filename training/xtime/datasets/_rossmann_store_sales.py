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
import calendar
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from xtime.ml import Feature, FeatureType, RegressionTask

from .dataset import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from .preprocessing import ChangeColumnsTypeToCategory, CheckColumnsOrder, DropColumns, EncodeCategoricalColumns

__all__ = ["RossmannStoreSalesBuilder"]

from ..errors import DatasetError


class RossmannStoreSalesBuilder(DatasetBuilder):
    NAME = "rossmann_store_sales"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(
            default=self._build_default_dataset,
            numerical=self._build_numerical_dataset,
            numerical32=functools.partial(self._build_numerical_dataset, precision="single"),
        )
        self._data_dir = Path("~/.cache/kaggle/datasets/rossmann_store_sales").expanduser()
        self._train_file = "train.csv.gz"
        self._store_file = "store.csv.gz"

    def _check_pre_requisites(self) -> None:
        if not ((self._data_dir / self._train_file).is_file() and (self._data_dir / self._store_file).is_file()):
            raise DatasetError.missing_prerequisites(
                f"Rossmann store sales dataset not found. Please download it from "
                f"`https://www.kaggle.com/competitions/rossmann-store-sales` and extract to "
                f"{self._data_dir.as_posix()}. Then, uncompress the archive and compress individual files with gzip "
                f"tool. To proceed, these files must exist: {(self._data_dir / self._train_file).as_posix()} and "
                f"{(self._data_dir / self._store_file).as_posix()}."
            )

    def _build_default_dataset(self, **kwargs) -> Dataset:
        train: pd.DataFrame = pd.read_csv((self._data_dir / self._train_file).as_posix())
        store: pd.DataFrame = pd.read_csv((self._data_dir / self._store_file).as_posix())

        # https://docs.python.org/3/library/calendar.html#calendar.month_abbr
        month_abbrs = list(calendar.month_abbr[1:])
        # It's `Sep` by default, but dataset uses Sept.
        month_abbrs[8] = "Sept"

        # StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state
        # holidays. Note that all schools are closed on public holidays and weekends. a = public holiday,
        # b = Easter holiday, c = Christmas, 0 = None
        train["StateHoliday"].replace(0, "n", inplace=True)

        # Convert Date column (e.g., 2015-07-31) into three integer columns - year, month and day
        train[["Year", "Month", "Day"]] = train["Date"].str.split(pat="-", n=3, expand=True).astype(int)
        train.drop(["Date"], axis=1, inplace=True)

        # Join with store table
        train = train.join(store, on="Store", rsuffix="_right")
        train.drop(["Store_right"], axis=1, inplace=True)

        # Convert `PromoInterval` (e.g., Jan,Apr,Jul,Oct) into binary variables
        promo2_start_months = [(s.split(",") if not pd.isnull(s) else []) for s in train["PromoInterval"]]
        for month_abbr in month_abbrs:
            train["Promo2Start_" + month_abbr] = np.array([(1 if month_abbr in s else 0) for s in promo2_start_months])
        train.drop(["PromoInterval"], axis=1, inplace=True)

        # StoreType - differentiates between 4 different store models: a, b, c, d
        train["StoreType"].fillna("na", inplace=True)
        # Assortment - describes an assortment level: a = basic, b = extra, c = extended
        train["Assortment"].fillna("na", inplace=True)

        # CompetitionDistance - distance in meters to the nearest competitor store
        train["CompetitionDistance"].fillna(-1, inplace=True)
        train["CompetitionOpenSinceMonth"].fillna(0, inplace=True)
        train["CompetitionOpenSinceYear"].fillna(0, inplace=True)

        # Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating,
        # 1 = store is participating
        train["Promo2"].fillna(0, inplace=True)
        train["Promo2SinceWeek"].fillna(-1, inplace=True)
        train["Promo2SinceYear"].fillna(-1, inplace=True)

        train["Promo2"] = train["Promo2"].astype(int)

        # Split into train/test splits
        train_indices = train[train["Year"] == 2014].index
        test_indices = train[train["Year"] == 2015].index

        train_split = train.iloc[train_indices].reset_index(drop=True)
        test_split = train.iloc[test_indices].reset_index(drop=True)

        label = "Sales"
        features = [
            Feature("Store", FeatureType.NOMINAL),
            Feature("DayOfWeek", FeatureType.NOMINAL),
            Feature("Customers", FeatureType.CONTINUOUS),
            Feature("Open", FeatureType.BINARY),
            Feature("Promo", FeatureType.BINARY),
            Feature("StateHoliday", FeatureType.NOMINAL),
            Feature("SchoolHoliday", FeatureType.BINARY),
            Feature("Month", FeatureType.CONTINUOUS),
            Feature("Day", FeatureType.CONTINUOUS),
            Feature("StoreType", FeatureType.NOMINAL),
            Feature("Assortment", FeatureType.NOMINAL),
            Feature("CompetitionDistance", FeatureType.CONTINUOUS),
            Feature("CompetitionOpenSinceMonth", FeatureType.CONTINUOUS),
            Feature("CompetitionOpenSinceYear", FeatureType.CONTINUOUS),
            Feature("Promo2", FeatureType.BINARY),
            Feature("Promo2SinceWeek", FeatureType.CONTINUOUS),
            Feature("Promo2SinceYear", FeatureType.CONTINUOUS),
        ]
        for month_abbr in month_abbrs:
            features.append(Feature(f"Promo2Start_{month_abbr}", FeatureType.BINARY))

        pipeline = Pipeline(
            [
                # Remove 'Year' column since it's irrelevant here
                ("drop_cols", DropColumns(["Year"])),
                # These are `object` columns (strings)
                ("cat_encoder", EncodeCategoricalColumns(["StateHoliday", "StoreType", "Assortment"])),
                # Update col types for binary features
                ("set_category_type", ChangeColumnsTypeToCategory(features)),
                # Check columns are in the right order
                ("check_cols_order", CheckColumnsOrder([f.name for f in features], label=label)),
            ]
        )
        train_split = pipeline.fit_transform(train_split)
        test_split = pipeline.transform(test_split)

        assert len(features) == train_split.shape[1] - 1, "Check 1 failed."
        assert len(features) == test_split.shape[1] - 1, "Check 2 failed."

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=RossmannStoreSalesBuilder.NAME,
                version="default",
                task=RegressionTask(),
                features=features,
                properties={"source": self._data_dir.as_uri()},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(
                    x=train_split.drop(label, axis=1, inplace=False), y=train_split[label]
                ),
                DatasetSplit.TEST: DatasetSplit(x=test_split.drop(label, axis=1, inplace=False), y=test_split[label]),
            },
        )
        return dataset
