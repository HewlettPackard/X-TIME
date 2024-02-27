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

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

from .dataset import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from .preprocessing import ChangeColumnsTypeToCategory, CheckColumnsOrder

__all__ = ["TelcoCustomerChurnBuilder"]

from ..errors import DatasetError


class TelcoCustomerChurnBuilder(DatasetBuilder):
    NAME = "telco_customer_churn"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset, numerical=self._build_numerical_dataset)
        self._data_dir = Path("~/.cache/kaggle/datasets/blastchar").expanduser()
        self._data_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    def _check_pre_requisites(self) -> None:
        if not (self._data_dir / self._data_file).is_file():
            raise DatasetError.missing_prerequisites(
                f"Blastchar (Telco Customer Churn) not found. Please download it from "
                f"`https://www.kaggle.com/datasets/blastchar/telco-customer-churn` and extract to "
                f"{self._data_dir.as_posix()}. To proceed, this file "
                f"must exist: {(self._data_dir / self._data_file).as_posix()}."
            )

    def _build_default_dataset(self) -> Dataset:
        """Create `blastchar (Telco Customer Churn)` train/valid/test datasets.

            Dataset source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
            Preprocessing pipeline same as in: https://arxiv.org/abs/2106.03253
            Paper loss: 0.2039
            Obtained test loss: mean=0.41119095600237765, std=0.00299323953304538
        Predict behavior to retain customers (customers who left within the last month – the column is called Churn).
            Size: 7,032 examples
            Input: 19 features
            Task: binary classification
        """
        # Load data
        data: pd.DataFrame = pd.read_csv((self._data_dir / self._data_file).as_posix())

        # Pretty much all fields are categorical, except `customerID`. This one needs to be removed.
        data.drop("customerID", axis=1, inplace=True)

        # This field has type int: convert to float
        data["tenure"] = data["tenure"].astype(float)

        # This field is object, convert to floating point numbers and remove nans
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
        data.dropna(axis=0, how="any", inplace=True)

        # Process categorical columns (I think some of these are actually binary columns)
        # binary 0/1 (Churn - label)
        data["gender"] = LabelEncoder().fit_transform(data["gender"])
        for feature in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]:
            data[feature].replace({"No": 0, "Yes": 1}, inplace=True)

        # categorical 0/1/2 or 0/1/2/3/4 (PaymentMethod)
        for feature in [
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaymentMethod",
        ]:
            data[feature] = LabelEncoder().fit_transform(data[feature])

        features = []
        for feature in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
            features.append(Feature(feature, FeatureType.BINARY))
        features.append(Feature("tenure", FeatureType.CONTINUOUS, cardinality=int(data["tenure"].nunique())))
        features.append(Feature("PhoneService", FeatureType.BINARY))
        for feature in [
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
        ]:
            features.append(Feature(feature, FeatureType.NOMINAL, cardinality=3))
        features.append(Feature("PaperlessBilling", FeatureType.BINARY))
        features.append(Feature("PaymentMethod", FeatureType.NOMINAL, cardinality=4))
        for feature in ["MonthlyCharges", "TotalCharges"]:
            features.append(Feature(feature, FeatureType.CONTINUOUS, cardinality=int(data[feature].nunique())))

        label: str = "Churn"
        pipeline = Pipeline(
            [
                # Check columns are in the right order
                ("check_cols_order", CheckColumnsOrder([f.name for f in features], label=label)),
                # Update col types for binary features
                ("set_category_type", ChangeColumnsTypeToCategory(features)),
            ]
        )
        data = pipeline.fit_transform(data)

        # In the paper, they do 80/20 stratified random split -> valid/test splits are going to be the same here.
        train, test = train_test_split(data, train_size=0.8, random_state=0, stratify=data[label])

        # Dataset (problem - multi-class classification, no categorical features present)
        dataset = Dataset(
            metadata=DatasetMetadata(
                name=TelcoCustomerChurnBuilder.NAME,
                version="default",
                task=ClassificationTask(TaskType.BINARY_CLASSIFICATION, num_classes=2),
                features=features,
                properties={"source": (self._data_dir / self._data_file).as_uri()},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(x=train.drop(label, axis=1, inplace=False), y=train[label]),
                DatasetSplit.TEST: DatasetSplit(x=test.drop(label, axis=1, inplace=False), y=test[label]),
            },
        )
        return dataset
