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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

from xtime.estimators.estimator import LegacySavedModelInfo, Model
from xtime.ml import TaskType

__all__ = ["get_model_stats"]


def get_model_stats(model_path: Path, task_type: TaskType) -> t.Dict:
    """Compute some basic statistics of a mode.

    Args:
        model_path: Directory that contains a Scikit-Learn random forest model.
        task_type: Task type this model was trained for.

    Returns:
        Dictionary with some descriptive statistics of this model.
    """
    model: t.Union[RandomForestClassifier, RandomForestRegressor] = Model.load_model(
        model_path, LegacySavedModelInfo("rf", task_type.value)
    )
    if not isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        raise ValueError(f"Unexpected ScikitLearn model loaded from '{model_path}' (type = {type(model)}).")

    if len(model.estimators_) > 0:
        if not isinstance(model.estimators_[0], DecisionTreeClassifier):
            raise ValueError(
                f"Unexpected ScikitLearn model loaded from '{model_path}' (estimator_type = {type(model.estimator_)})."
            )

    max_depth, max_leaves = 0, 0
    for estimator in model.estimators_:
        max_depth = max(max_depth, estimator.get_depth())
        max_leaves = max(max_leaves, estimator.get_n_leaves())

    return {"max_depth": max_depth, "max_leaves": max_leaves, "num_trees": len(model.estimators_)}
