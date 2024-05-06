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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

from .dataset import Dataset, DatasetBuilder, DatasetMetadata, DatasetPrerequisites, DatasetSplit
from .preprocessing import ChangeColumnsType, ChangeColumnsTypeToCategory, CheckColumnsOrder, DropColumns

__all__ = ["EyeMovementsBuilder"]


class EyeMovementsBuilder(DatasetBuilder):
    NAME = "eye_movements"

    def __init__(self) -> None:
        super().__init__(openml=True)
        self.builders.update(
            default=self._build_default_dataset,
            numerical=self._build_numerical_dataset,
            numerical32=functools.partial(self._build_numerical_dataset, precision="single"),
        )

    def _check_pre_requisites(self) -> None:
        DatasetPrerequisites.check_openml(self.NAME, "openml.org/d/1044")

    def _build_default_dataset(self, **kwargs) -> Dataset:
        """Create `eye_movements` train/valid/test datasets.

            Dataset source: openml.org/d/1044
            Preprocessing pipeline same as in: https://arxiv.org/abs/2106.03253
            Paper loss: 0.5607
            Obtained test loss: mean=0.6008637686347453, std=0.006451555003031239
        http://research.ics.aalto.fi/events/eyechallenge2005/irem-2005-03-03.pdf
        http://research.ics.aalto.fi/events/eyechallenge2005/eyechallenge2005poster.pdf
        In information retrieval, relevance generally depends on the context, task, and individual competence and
        preferences of the user. Therefore, relevance of articles suggested by a search engine could be improved by
        filtering them through an algorithm which models the interests of the user. The Challenge is to predict from eye
        movement data whether a reader finds a text relevant.
            Size: 10,936 examples
            Input: 26 features
            Task: multi-class classification (Irrelevant / Relevant / Correct)
        """
        from openml.datasets import get_dataset as get_openml_dataset
        from openml.datasets.dataset import OpenMLDataset

        # Init parameters.
        random_state: int = 0
        validation_size: float = 0.1
        test_size: float = 0.2

        # Fetch dataset and its description from OpenML. Will be cached in ${HOME}/.openml
        data: OpenMLDataset = get_openml_dataset(
            dataset_id="eye_movements", version=1, error_if_multiple=True, download_data=True
        )

        # Load from local cache (x - pandas data frame, y - pandas series)
        x, y, _, _ = data.get_data(target=data.default_target_attribute, dataset_format="dataframe")
        assert isinstance(x, pd.DataFrame), f"Expecting x to be of type pd.DataFrame (type = {type(x)})."
        assert isinstance(y, pd.Series), f"Expecting y to be of type pd.Series (type = {type(y)})."

        # Encode labels. Move from `category` type to int type with labels [0, 1, 2]
        y = y.astype(int)

        #
        _binary_features = ["P1stFixation", "P2stFixation", "nextWordRegress"]
        _drop_features = ["lineNo"]
        features = []
        for feature in x.columns:
            if feature in _drop_features:
                continue
            if feature in _binary_features:
                features.append(Feature(feature, FeatureType.BINARY))
            else:
                features.append(Feature(feature, FeatureType.CONTINUOUS, cardinality=int(x[feature].nunique())))

        # Pipeline to pre-process data by removing unused columns and fixing data types.
        pipeline = Pipeline(
            [
                # Drop unique columns
                ("drop_cols", DropColumns(["lineNo"])),
                # Convert `category` features to int type (no need to encode them - already 0/1.).
                ("change_cols_type", ChangeColumnsType(_binary_features, dtype=int)),
                # Check columns are in the right order
                ("check_cols_order", CheckColumnsOrder([f.name for f in features])),
                # Update col types for binary features
                ("set_category_type", ChangeColumnsTypeToCategory(features)),
            ]
        )
        x = pipeline.fit_transform(x)

        # Remove instances with missing values
        # No need to remove instances with missing values - no such instances.

        # Split into train/valid/test according to paper
        train_x, test_x, train_y, test_y = train_test_split(
            x, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        train_x, valid_x, train_y, valid_y = train_test_split(
            train_x, train_y, test_size=validation_size / (1.0 - test_size), random_state=random_state, shuffle=True
        )

        # Return dataset (problem - multi-class classification, no categorical features present)
        dataset = Dataset(
            metadata=DatasetMetadata(
                name=EyeMovementsBuilder.NAME,
                version="default",
                task=ClassificationTask(TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=3),
                features=features,
                properties={"source": data.data_file},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(x=train_x, y=train_y),
                DatasetSplit.VALID: DatasetSplit(x=valid_x, y=valid_y),
                DatasetSplit.TEST: DatasetSplit(x=test_x, y=test_y),
            },
        )
        return dataset
