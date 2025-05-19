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

import math
import typing as t

from ray import tune
from tinydb import Query, TinyDB
from tinydb.storages import MemoryStorage

from xtime.ml import TaskType

from ._hparams import HParams, HParamsSpace, HParamsSpec, ValueSpec

_CLS_TASKS = {TaskType.BINARY_CLASSIFICATION, TaskType.MULTI_CLASS_CLASSIFICATION}
_REG_TASKS = {TaskType.REGRESSION}
_ALL_TASKS = {TaskType.BINARY_CLASSIFICATION, TaskType.MULTI_CLASS_CLASSIFICATION, TaskType.REGRESSION}


class DefaultRecommender(object):
    def __init__(self) -> None:
        self._db: TinyDB = TinyDB(storage=MemoryStorage)
        self._db.insert_multiple(
            [
                {
                    "tags": {"model": "lightgbm", "tasks": _ALL_TASKS},
                    "params": HParamsSpec(
                        n_estimators=ValueSpec(int, 100, tune.randint(100, 4001)),
                        learning_rate=ValueSpec(float, 0.3, tune.loguniform(1e-7, 1)),
                        max_depth=ValueSpec(int, 6, tune.randint(1, 11)),
                        colsample_bytree=ValueSpec(float, 1, tune.uniform(0.2, 1)),
                        reg_alpha=ValueSpec(float, 0, tune.loguniform(math.exp(-16), math.exp(2))),
                        reg_lambda=ValueSpec(float, 1, tune.loguniform(math.exp(-16), math.exp(2))),
                        random_state=ValueSpec(int, 1, 1),
                    ),
                },
                {
                    "tags": {"model": "dummy", "tasks": _CLS_TASKS},
                    "params": HParamsSpec(strategy=ValueSpec(str, "prior", "prior"), random_state=ValueSpec(int, 1, 1)),
                },
                {
                    "tags": {"model": "dummy", "tasks": _REG_TASKS},
                    "params": HParamsSpec(strategy=ValueSpec(str, "mean", "mean")),
                },
                {
                    "tags": {"model": "rf", "tasks": _ALL_TASKS},
                    "params": HParamsSpec(
                        n_estimators=ValueSpec(int, 100, tune.randint(100, 4001)),
                        max_depth=ValueSpec(int, 6, tune.randint(1, 11)),
                        random_state=ValueSpec(int, 1, 1),
                    ),
                },
                {
                    "tags": {"model": "catboost", "tasks": _ALL_TASKS},
                    "params": HParamsSpec(
                        learning_rate=ValueSpec(float, 0.03, tune.loguniform(0.00001, 1)),
                        random_strength=ValueSpec(float, 1, tune.choice(range(1, 21))),
                        depth=ValueSpec(int, 6, tune.choice(range(1, 17))),
                        l2_leaf_reg=ValueSpec(float, 3, tune.loguniform(1, 10)),
                        bagging_temperature=ValueSpec(float, 1, tune.uniform(0, 1)),
                        leaf_estimation_iterations=ValueSpec(int, 1, tune.choice(range(1, 21))),
                        random_state=ValueSpec(int, 1, 1),
                    ),
                },
                {
                    "tags": {"model": "xgboost", "tasks": _ALL_TASKS},
                    "params": HParamsSpec(
                        n_estimators=ValueSpec(int, 100, tune.randint(100, 4001)),
                        learning_rate=ValueSpec(float, 0.3, tune.loguniform(1e-7, 1)),
                        max_depth=ValueSpec(int, 6, tune.randint(1, 11)),
                        subsample=ValueSpec(float, 1, tune.uniform(0.2, 1)),
                        colsample_bytree=ValueSpec(float, 1, tune.uniform(0.2, 1)),
                        colsample_bylevel=ValueSpec(float, 1, tune.uniform(0.2, 1)),
                        min_child_weight=ValueSpec(float, 1, tune.loguniform(math.exp(-16), math.exp(5))),
                        reg_alpha=ValueSpec(float, 0, tune.loguniform(math.exp(-16), math.exp(2))),
                        reg_lambda=ValueSpec(float, 1, tune.loguniform(math.exp(-16), math.exp(2))),
                        gamma=ValueSpec(float, 0, tune.loguniform(math.exp(-16), math.exp(2))),
                        random_state=ValueSpec(int, 1, 1),
                    ),
                },
            ]
        )

    def recommend(self, query: Query) -> t.List[HParamsSpec]:
        """Recommend hyperparameters for a given query.

        Args:
            query: Query to search for. The following is the query example:
            >>> q = Query()
            ... _ = self.recommend(q.tags.model == 'lightgbm' and q.tags.tasks.any([TaskType.BINARY_CLASSIFICATION]))
        Returns:
            List of hyperparameter specifications.
        """
        matches: t.List[t.Dict] = self._db.search(query)
        return [match["params"] for match in matches]

    def recommend_default_values(self, query: Query) -> t.List[HParams]:
        matches: t.List[HParamsSpec] = self.recommend(query)
        return [match.default() for match in matches]

    def recommend_search_space(self, query: Query) -> t.List[HParamsSpace]:
        matches: t.List[HParamsSpec] = self.recommend(query)
        return [match.space() for match in matches]
