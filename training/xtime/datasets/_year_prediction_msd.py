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

import logging
from pathlib import Path

import pandas as pd

from xtime.io import IO
from xtime.ml import Feature, FeatureType, RegressionTask

from .dataset import Dataset, DatasetBuilder, DatasetMetadata, DatasetSplit

__all__ = ["YearPredictionMSDBuilder"]

logger = logging.getLogger(__name__)


class YearPredictionMSDBuilder(DatasetBuilder):
    NAME = "year_prediction_msd"

    def __init__(self) -> None:
        super().__init__()
        self.builders.update(default=self._build_default_dataset)

    def _build_default_dataset(self, **kwargs) -> Dataset:
        """

        https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
        https://intel.github.io/scikit-learn-intelex/samples/linear_regression.html
        Prediction of the release year of a song from audio features. Songs are mostly western, commercial tracks
        ranging from 1922 to 2011, with a peak in the year 2000s.
            Size: 515,345 examples
            Input: 90 features
            Task: regression
        """
        data_dir = Path("~/.cache/uci/datasets/00203").expanduser()
        file_name = "YearPredictionMSD.txt.zip"
        if not (data_dir / file_name).is_file():
            logger.debug("Downloading Year Prediction dataset to: %s.", (data_dir / file_name).as_posix())
            IO.download(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip",
                data_dir,
                file_name,
            )

        # Load data
        data: pd.DataFrame = pd.read_csv((data_dir / file_name).as_posix())

        label = "Year"

        # Dataset comes without columns. 90 attributes, 12 = timbre average, 78 = timbre covariance
        # The first value is the year (target), ranging from 1922 to 2011.
        columns = ["Year"]
        for i in range(12):
            columns.append(f"TimbreAvg_{i}")
        for i in range(78):
            columns.append(f"TimbreCov_{i}")
        assert len(columns) == 91, f"Fix me. Length = {len(columns)}"
        data.columns = columns

        # No need to update column types - all features are continuous

        # According to dataset [documentation](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd):
        # You should respect the following train / test split:
        #    - train: first 463,715 examples
        #    - test: last 51,630 examples
        # It avoids the 'producer effect' by making sure no song from a given artist ends up in both the train
        # and test set.
        train: pd.DataFrame = data.iloc[0:463715, :]
        test: pd.DataFrame = data.iloc[-51630:, :]

        dataset = Dataset(
            metadata=DatasetMetadata(
                name=YearPredictionMSDBuilder.NAME,
                version="default",
                task=RegressionTask(),
                features=[Feature(name, FeatureType.CONTINUOUS) for name in columns[1:]],
                properties={"source": (data_dir / file_name).as_uri()},
            ),
            splits={
                DatasetSplit.TRAIN: DatasetSplit(x=train.drop(label, axis=1, inplace=False), y=train[label]),
                DatasetSplit.TEST: DatasetSplit(x=test.drop(label, axis=1, inplace=False), y=test[label]),
            },
        )
        return dataset
