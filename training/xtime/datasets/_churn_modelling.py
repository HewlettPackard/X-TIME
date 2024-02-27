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
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xtime.io import IO
from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

from .dataset import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from .preprocessing import (
    ChangeColumnsType,
    ChangeColumnsTypeToCategory,
    CheckColumnsOrder,
    DropColumns,
    EncodeCategoricalColumns,
)

__all__ = ["ChurnModellingBuilder"]

from ..errors import DatasetError


class ChurnModellingBuilder(DatasetBuilder):
    NAME = "churn_modelling"

    def __init__(self, path: t.Optional[t.Union[str, Path]] = None, **kwargs) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset, numerical=self._build_numerical_dataset)
        self.path: Path = IO.get_path(path, "~/.cache/kaggle/datasets/shrutime")
        self.file_name = kwargs.get("file_name", "Churn_Modelling.csv")

    def _check_pre_requisites(self) -> None:
        if not (self.path / self.file_name).is_file():
            raise DatasetError.missing_prerequisites(
                f"Shrutime (Telco Modelling) not found. Please download it from "
                f"`https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling` and extract to "
                f"{self.path.as_posix()}. To proceed, this file must exist: {(self.path / self.file_name).as_posix()}"
            )

    def _build_default_dataset(self, **kwargs) -> Dataset:
        """Create `shrutime (Telco Modelling)` train/valid/test datasets.

            Dataset source: https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
            Preprocessing pipeline same as in: https://arxiv.org/abs/2106.03253
            Paper loss: 0.1382
            Obtained test loss: mean=0.3410263324187369, std=0.0018547527673419342
        This data set contains details of a bank's customers and the target variable is a binary variable reflecting the
        fact whether the customer left the bank (closed his account) or he continues to be a customer.
            Size: 10,000 examples
            Input: 10 features
            Task: binary classification
        """
        if kwargs:
            raise ValueError(f"{self.__class__.__name__}: `default` dataset does not accept arguments.")
        # Load data
        data_path = self.path / self.file_name
        data: pd.DataFrame = pd.read_csv(data_path.as_posix())

        # These are the features this dataset provides
        features = [
            Feature("CreditScore", FeatureType.CONTINUOUS),
            Feature("Geography", FeatureType.NOMINAL),
            Feature("Gender", FeatureType.BINARY),
            Feature("Age", FeatureType.CONTINUOUS),
            Feature("Tenure", FeatureType.CONTINUOUS),
            Feature("Balance", FeatureType.CONTINUOUS),
            Feature("NumOfProducts", FeatureType.ORDINAL),
            Feature("HasCrCard", FeatureType.BINARY),
            Feature("IsActiveMember", FeatureType.BINARY),
            Feature("EstimatedSalary", FeatureType.CONTINUOUS),
        ]

        # Pipeline to pre-process data by removing unused columns and fixing data types.
        label: str = "Exited"
        pipeline = Pipeline(
            [
                # Drop unique columns ('RowNumber', 'CustomerId') and textual fields (second names - 'Surname')
                ("drop_cols", DropColumns(["RowNumber", "CustomerId", "Surname"])),
                # Convert several numerical columns to floating point format
                ("change_cols_type", ChangeColumnsType(["CreditScore", "Age", "Tenure"], dtype=float)),
                # Check columns are in the right order
                ("check_cols_order", CheckColumnsOrder([f.name for f in features], label=label)),
            ]
        )
        data = pipeline.fit_transform(data)

        # In the paper, they do 80/20 stratified random split -> valid/test splits are going to be the same here.
        train, test = train_test_split(data, train_size=0.8, random_state=0, stratify=data[label])

        # Pipeline to encode categorical features
        pipeline = Pipeline(
            [
                ("cat_encoder", EncodeCategoricalColumns(["Geography", "Gender"])),
                ("set_category_type", ChangeColumnsTypeToCategory(features)),
            ]
        )
        train = pipeline.fit_transform(train)
        test = pipeline.transform(test)

        # Return dataset (problem - multi-class classification, no categorical features present)
        dataset = Dataset(
            metadata=DatasetMetadata(
                name=ChurnModellingBuilder.NAME,
                version="default",
                task=ClassificationTask(TaskType.BINARY_CLASSIFICATION, num_classes=2),
                features=features,
                properties={"source": data_path.as_uri()},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(x=train.drop(label, axis=1, inplace=False), y=train[label]),
                DatasetSplit.TEST: DatasetSplit(x=test.drop(label, axis=1, inplace=False), y=test[label]),
            },
        )
        return dataset
