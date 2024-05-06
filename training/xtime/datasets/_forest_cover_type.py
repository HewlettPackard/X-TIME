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
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import get_data_home
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch

from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

from .dataset import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit
from .preprocessing import ChangeColumnsTypeToCategory, CheckColumnsOrder

__all__ = ["ForestCoverTypeBuilder"]


class ForestCoverTypeBuilder(DatasetBuilder):
    NAME = "forest_cover_type"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(
            default=self._build_default_dataset,
            numerical=self._build_numerical_dataset,
            numerical32=functools.partial(self._build_numerical_dataset, precision="single"),
        )

    def _build_default_dataset(self, **kwargs) -> Dataset:
        """Create `Forest Cover Type (cover_type)` train/valid/test datasets.

            Dataset source: https://www.kaggle.com/c/forest-cover-type-prediction
              See also:
                https://github.com/RAMitchell/GBM-Benchmarks
                https://github.com/RAMitchell/GBM-Benchmarks/blob/a0bbed08c918b0a82e9a5e2207d1f43134b445e0/benchmark.py#L150
            Preprocessing pipeline same as in: https://arxiv.org/abs/2106.03253 (original paper points to GBM-... repo)
            Paper loss: 0.0313
            Obtained test loss: mean=0.08348248213805014, std=0.0002354895559675744
        https://www.kaggle.com/c/forest-cover-type-prediction:
        In this competition you are asked to predict the forest cover type (the predominant kind of tree cover) from
        strictly cartographic variables (as opposed to remotely sensed data). The actual forest cover type for a given
        30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data.
        Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is
        in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as
        wilderness areas and soil type.
        This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado.
        These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are
        more a result of ecological processes rather than forest management practices.
            Size: 581,012 examples
            Input: 54 features
            Task: multi-class classification - 7 classes (Spruce/Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow,
                Aspen, Douglas-fir, Krummholz)
        """
        # Load data - referenced implementation uses sklearn to fetch this dataset.
        bunch: Bunch = datasets.fetch_covtype(download_if_missing=True)
        data = pd.DataFrame(
            np.hstack([bunch.data, bunch.target.reshape((-1, 1))]), columns=bunch.feature_names + bunch.target_names
        )

        # In this dataset two types of features are present - continuous and binary. It seems that the binary features
        # are one-hot encoded features for original categorical features - Wilderness_Area (4 values) and
        # Soil_Type (40 values). Originally, all features have `float` data type.
        features = []
        label: str = "Cover_Type"
        for feature in data.columns:
            if feature.startswith("Wilderness_Area") or feature.startswith("Soil_Type"):
                data[feature] = data[feature].astype(int)
                features.append(Feature(feature, FeatureType.BINARY))
            elif feature == label:
                data[feature] = LabelEncoder().fit_transform(data[feature].astype(int))
            else:
                features.append(Feature(feature, FeatureType.CONTINUOUS, cardinality=int(data[feature].nunique())))

        pipeline = Pipeline(
            [
                # Check columns are in the right order
                ("check_cols_order", CheckColumnsOrder([f.name for f in features], label=label)),
                # Update col types for binary features
                ("set_category_type", ChangeColumnsTypeToCategory(features)),
            ]
        )
        data = pipeline.fit_transform(data)

        # Split datasets
        # https://github.com/RAMitchell/GBM-Benchmarks/blob/a0bbed08c918b0a82e9a5e2207d1f43134b445e0/benchmark.py#L150
        test_size = 0.2
        validation_size = 0.2

        train, test = train_test_split(data, test_size=test_size, random_state=0)
        train, valid = train_test_split(train, test_size=validation_size / (1.0 - test_size), random_state=0)

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=ForestCoverTypeBuilder.NAME,
                version="default",
                task=ClassificationTask(TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=7),
                features=features,
                properties={"source": (Path(get_data_home()) / "covertype").as_uri()},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(x=train.drop(label, axis=1, inplace=False), y=train[label]),
                DatasetSplit.VALID: DatasetSplit(x=valid.drop(label, axis=1, inplace=False), y=valid[label]),
                DatasetSplit.TEST: DatasetSplit(x=test.drop(label, axis=1, inplace=False), y=test[label]),
            },
        )
        return dataset
