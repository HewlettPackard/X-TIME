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

import copy
import typing as t
from dataclasses import dataclass, field
from enum import Enum

from xtime.datasets import Dataset

__all__ = ["RunType", "Metadata", "Context"]


class RunType(str, Enum):
    """Type of run."""

    HPO = "hpo"
    """Hyperparameter optimization run."""

    TRAIN = "train"
    """Training run."""


@dataclass
class Metadata:
    """Metadata for a run (part of a more general context).

    It could be used to pass datasets and additional parameters (that are not hyperparameters) to fit functions.
    """

    dataset: str
    model: str
    run_type: RunType
    fit_params: t.Dict = field(default_factory=dict)

    def to_json(self) -> t.Dict:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "run_type": self.run_type.value,
            "fit_params": copy.deepcopy(self.fit_params),
        }


@dataclass
class Context:
    """Context for a run."""

    metadata: Metadata
    dataset: t.Optional[Dataset] = None
    callbacks: t.Optional[t.List["Callback"]] = None
