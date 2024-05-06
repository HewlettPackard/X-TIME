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
from sklearn.preprocessing import LabelEncoder

from xtime.ml import ClassificationTask, Feature, FeatureType, TaskType

from .dataset import Dataset, DatasetBuilder, DatasetMetadata, DatasetPrerequisites, DatasetSplit

__all__ = ["GesturePhaseSegmentationBuilder"]


class GesturePhaseSegmentationBuilder(DatasetBuilder):
    NAME = "gesture_phase_segmentation"

    def __init__(self) -> None:
        super().__init__(openml=True)
        self.builders.update(
            default=self._build_default_dataset,
            numerical=self._build_numerical_dataset,
            numerical32=functools.partial(self._build_numerical_dataset, precision="single"),
        )

    def _check_pre_requisites(self) -> None:
        DatasetPrerequisites.check_openml(self.NAME, "openml.org/d/4538")

    def _build_default_dataset(self, **kwargs) -> Dataset:
        """Create `Gesture Phase Segmentation Processed` train/valid/test datasets.

            Dataset source: openml.org/d/4538
            Preprocessing pipeline same as in: https://arxiv.org/abs/2106.03253
            Paper loss: 0.8064
            Obtained test loss: mean=0.8009250719858734, std=0.003496130752670844
        The dataset is composed by features extracted from 7 videos with people gesticulating, aiming at studying
        Gesture Phase Segmentation
            Size: 9,873 examples
            Input: 32 features
            Task: multi-class classification - 5 classes (RestPosition, Preparation, Stroke, Hold, Retraction).
        """
        from openml.datasets import get_dataset as get_openml_dataset
        from openml.datasets.dataset import OpenMLDataset

        # Init parameters.
        random_state: int = 0
        validation_size: float = 0.1
        test_size: float = 0.2

        # Fetch dataset and its description from OpenML. Will be cached in ${HOME}/.openml
        data: OpenMLDataset = get_openml_dataset(
            dataset_id="GesturePhaseSegmentationProcessed", version=1, error_if_multiple=True, download_data=True
        )

        # Load from local cache
        # features               pd.DataFrame  (9873, 32)
        # labels                 pd.Series     (9873, )
        # categorical_indicator  t.List[bool]  32          No categorical features
        # attributed_names       t.List[str]   32          5 categories: ['D', 'P', 'S', 'H', 'R']
        x, y, _, _ = data.get_data(target=data.default_target_attribute, dataset_format="dataframe")
        assert isinstance(x, pd.DataFrame), f"Expecting x to be of type pd.DataFrame (type = {type(x)})."
        assert isinstance(y, pd.Series), f"Expecting y to be of type pd.Series (type = {type(y)})."

        # Encode labels
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index, name=y.name)

        # Remove instances with missing values
        # No need to remove instances with missing values - no such instances.

        # All features in this dataset are continuous (float64)
        features = [Feature(col, FeatureType.CONTINUOUS, cardinality=int(x[col].nunique())) for col in x.columns]

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
                name=GesturePhaseSegmentationBuilder.NAME,
                version="default",
                task=ClassificationTask(TaskType.MULTI_CLASS_CLASSIFICATION, num_classes=5),
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
