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
from enum import Enum, unique

__all__ = ["TaskType", "Task", "ClassificationTask", "RegressionTask", "FeatureType", "Feature", "METRICS"]


@unique
class TaskType(str, Enum):
    """Type of optimization task."""

    BINARY_CLASSIFICATION = "binary_classification"
    """Binary classification task."""

    MULTI_CLASS_CLASSIFICATION = "multi_class_classification"
    """Multi-class classification task."""

    REGRESSION = "regression"
    """Uni-variate regression task."""

    def classification(self) -> bool:
        return self in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTI_CLASS_CLASSIFICATION)

    def regression(self) -> bool:
        return self is TaskType.REGRESSION


class Task:
    def __init__(self, type_: t.Union[TaskType, str]) -> None:
        self.type = TaskType(type_)

    def to_json(self) -> t.Dict:
        return {"type": self.type.value}

    @staticmethod
    def from_json(json_obj: t.Dict) -> "Task":
        json_obj = copy.deepcopy(json_obj)
        type_: TaskType = TaskType(json_obj.pop("type"))
        if type_.classification():
            cls = ClassificationTask
        elif type_.regression():
            cls = RegressionTask
        else:
            raise ValueError(f"Unrecognized task type ({type_})")
        return cls(type_, **json_obj)


class ClassificationTask(Task):
    def __init__(self, type_: t.Union[TaskType, str], num_classes: t.Optional[int] = None) -> None:
        type_ = TaskType(type_)
        assert type_.classification(), f"Not a classification task type ({type_})."

        if num_classes is None and type_ == TaskType.BINARY_CLASSIFICATION:
            num_classes = 2
        if num_classes is None or num_classes <= 0:
            raise ValueError(f"Number of classes must be a positive integer. Got {num_classes}.")
        if type_ == TaskType.BINARY_CLASSIFICATION and num_classes != 2:
            raise ValueError(
                f"Number of classes must equal to 2 when task is binary classification. Got {num_classes}."
            )
        if type_ == TaskType.MULTI_CLASS_CLASSIFICATION and num_classes <= 2:
            raise ValueError(
                f"Number of classes must be > 2 when task is multi-class classification. Got {num_classes}."
            )

        super().__init__(type_)
        self.num_classes: int = num_classes

    def to_json(self) -> t.Dict:
        _dict = super().to_json()
        _dict.update(num_classes=self.num_classes)
        return _dict


class RegressionTask(Task):
    def __init__(self, type_: t.Optional[t.Union[TaskType, str]] = None, **kwargs) -> None:
        assert len(kwargs) == 0, f"No additional parameters are supported ({kwargs})."
        type_ = TaskType(type_ or TaskType.REGRESSION)
        assert type_.regression(), f"Not a regression task type ({type_})."
        super().__init__(type_)


@unique
class FeatureType(str, Enum):
    # Quantitative (numerical) features
    DISCRETE = "discrete"
    """Any numbers of type float (32/64) taking only certain values."""

    CONTINUOUS = "continuous"
    """Any numbers of type float (32/64)."""

    # Qualitative (categorical) features
    ORDINAL = "ordinal"
    """Categories that maintain an order that are always of type int (32/64)."""

    NOMINAL = "nominal"
    """Categories with no order ranking that are always of type int (32/64) with values starting at 0.
    Nominal features with cardinality equal to 2 have a dedicated label - binary.
    """

    BINARY = "binary"
    """Nominal features with two categories that are always of type int (32/64) with values 0 and 1."""

    def numerical(self) -> bool:
        return self in (FeatureType.DISCRETE, FeatureType.CONTINUOUS)

    def categorical(self) -> bool:
        return self in (FeatureType.ORDINAL, FeatureType.NOMINAL, FeatureType.BINARY)

    def nominal(self) -> bool:
        return self in (FeatureType.NOMINAL, FeatureType.BINARY)


class Feature(object):
    def __init__(self, name: str, type_: t.Union[FeatureType, str], **kwargs) -> None:
        self.name = name
        self.type = FeatureType(type_)
        self.kwargs = copy.deepcopy(kwargs)

    def to_json(self) -> t.Dict:
        return {"name": self.name, "type": self.type.value, **self.kwargs}

    @classmethod
    def from_json(cls, json_dict: t.Dict) -> "Feature":
        json_dict = copy.deepcopy(json_dict)
        name, type_ = json_dict.pop("name"), json_dict.pop("type")
        return cls(name=name, type_=type_, **json_dict)

    def __str__(self) -> str:
        return f"Feature(name={self.name}, type={self.type}, kwargs={self.kwargs})"


class _Metrics:
    CLASSIFICATION = [
        "dataset_accuracy",
        "train_accuracy",
        "valid_accuracy",
        "test_accuracy",
        "dataset_loss_mean",
        "train_loss_mean",
        "valid_loss_mean",
        "test_loss_mean",
    ]
    REGRESSION = ["dataset_mse", "train_mse", "valid_mse", "test_mse"]

    def __init__(self) -> None:
        self._metrics = {
            TaskType.BINARY_CLASSIFICATION: _Metrics.CLASSIFICATION,
            TaskType.MULTI_CLASS_CLASSIFICATION: _Metrics.CLASSIFICATION,
            TaskType.REGRESSION: _Metrics.REGRESSION,
        }

    def __getitem__(self, key: TaskType) -> t.List[str]:
        if not isinstance(key, TaskType) or key not in self._metrics:
            raise KeyError(
                f"Invalid key (type={type(key)}, value={key}) in {self.__class__.__name__}. Expected keys of "
                f"type `TaskType` (valid_keys={list(self._metrics.keys())})."
            )
        return self._metrics[key]

    def __len__(self) -> int:
        return len(self._metrics)

    def to_json(self) -> t.Dict:
        return copy.deepcopy(self._metrics)

    @staticmethod
    def get_primary_metric(task: t.Union[Task, TaskType]) -> str:
        if isinstance(task, Task):
            task = task.type
        return "valid_mse" if task.regression() else "valid_loss_mean"


METRICS = _Metrics()
