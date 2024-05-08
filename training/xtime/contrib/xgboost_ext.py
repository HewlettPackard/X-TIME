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
import json
import typing as t
from pathlib import Path

import pandas as pd
from xgboost import XGBModel

from xtime.estimators.estimator import LegacySavedModelInfo, Model
from xtime.ml import TaskType

__all__ = ["TreeTraversal", "get_model_stats"]


class TreeTraversal:
    """Traverse dictionary representing one XGBoost tree and collect some descriptive statistics."""

    def __init__(self) -> None:
        self.depth = 0
        """Depth of a tree."""
        self.num_nodes = 0
        """Number of nodes in a tree (including leaves)."""
        self.num_leaves = 0
        """Number of leaves in a tree."""

    def traverse(self, node: t.Dict) -> "TreeTraversal":
        """Traverse the tree starting with `node` node."""
        self.depth = 0
        self.num_leaves = 0
        self.num_nodes = 0
        self._traverse(node)
        self.depth += 1  # Leaf nodes do not have `depth` field.
        return self

    def __repr__(self) -> str:
        return f"TreeTraversal(depth={self.depth}, num_leaves={self.num_leaves}, num_nodes={self.num_nodes})"

    def as_dict(self, prefix: str = "") -> t.Dict:
        return {
            f"{prefix}depth": self.depth,
            f"{prefix}num_leaves": self.num_leaves,
            f"{prefix}num_nodes": self.num_nodes,
        }

    def _traverse(self, node: t.Dict) -> None:
        """Recursively traverse the tree."""
        if not isinstance(node, dict) or "nodeid" not in node:
            return
        self.num_nodes += 1
        if "depth" in node:
            self.depth = max(self.depth, node["depth"])
        if "leaf" in node:
            self.num_leaves += 1
        if isinstance(node.get("children", None), list):
            for child in node["children"]:
                self._traverse(child)


def get_model_stats(
    model_dir: Path, task_type: TaskType, num_trees: int = 0, cache: t.Optional[t.Dict] = None
) -> t.Dict:
    """Compute some basic statistics of an XGBoost model.

    Args:
        model_dir: Directory that contains an XGBoost model.
        task_type: Task type this model was trained for (need to be able to load the right model).
        num_trees: Number of trees to use to compute statistics. For multi-class classification problems (num classes
            > 2), this must be `num_rounds * num_classes`. The value of `0` means use all trees.
        cache: If this function needs to be called multiple times, if the cache is not None, the function will cache
            the trees data frame in this cache under the "trees" key.
    Returns:
        Dictionary with some descriptive statistics of this model that include the following: maximal tree depth,
        maximal number of leaves, total number of trees. If `num_trees` is specified, and it is smaller or equal
        to actual number of trees, then number of trees will equal to this number.
    """

    def _get_stats(_trees: pd.DataFrame) -> t.Dict:
        _num_trees = _trees.shape[0] if num_trees <= 0 else min(num_trees, _trees.shape[0])
        return {
            "max_depth": _trees["depth"][0:_num_trees].max(),
            "max_leaves": _trees["num_leaves"][0:_num_trees].max(),
            "num_trees": _num_trees,
        }

    if cache is not None and "trees" in cache:
        return _get_stats(cache["trees"])

    if cache is not None and "model" in cache:
        model: XGBModel = cache["model"]
    else:
        model = Model.load_model(model_dir, LegacySavedModelInfo("xgboost", task_type.value))
    if not isinstance(model, XGBModel):
        raise ValueError(f"Unexpected XGBoost model loaded from '{model_dir}' (type = {type(model)}).")
    if cache is not None and "model" not in cache:
        cache["model"] = model

    # For multi-class classification problems, number of `trees = num_classes x num_trees_per_class`
    trees: pd.DataFrame = pd.DataFrame(
        [TreeTraversal().traverse(json.loads(n)).as_dict() for n in model.get_booster().get_dump(dump_format="json")]
    )
    if cache is not None:
        cache["trees"] = trees
    return _get_stats(trees)
