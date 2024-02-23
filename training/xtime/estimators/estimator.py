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

import abc
import copy
import os
import typing as t
from pathlib import Path
from unittest import TestCase

import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from xtime.datasets import Dataset, DatasetMetadata, DatasetSplit
from xtime.datasets.dataset import build_dataset
from xtime.io import IO, encode
from xtime.ml import METRICS, Task
from xtime.registry import ClassRegistry
from xtime.run import Context, Metadata, RunType

__all__ = ["Estimator", "get_estimator_registry", "get_estimator", "unit_test_train_model", "unit_test_check_metrics"]


class Callback(object):
    def before_fit(self, dataset: Dataset, estimator: "Estimator") -> None: ...

    def after_fit(self, dataset: Dataset, estimator: "Estimator") -> None: ...

    def after_test(self, dataset: Dataset, estimator: "Estimator", metrics: t.Dict) -> None: ...


class ContainerCallback(Callback):
    def __init__(self, callbacks: t.Optional[t.List[Callback]]) -> None:
        self.callbacks = callbacks or []

    def before_fit(self, dataset: Dataset, estimator: "Estimator") -> None:
        for callback in self.callbacks:
            callback.before_fit(dataset, estimator)

    def after_fit(self, dataset: Dataset, estimator: "Estimator") -> None:
        for callback in self.callbacks:
            callback.after_fit(dataset, estimator)

    def after_test(self, dataset: Dataset, estimator: "Estimator", metrics: t.Dict) -> None:
        for callback in self.callbacks:
            callback.after_test(dataset, estimator, metrics)


class MLflowCallback(Callback):
    def __init__(self, hparams: t.Dict, ctx: Context) -> None:
        from xtime.datasets.dataset import parse_dataset_name

        mlflow.log_params(hparams)

        dataset_name, dataset_version = parse_dataset_name(ctx.metadata.dataset)
        mlflow.set_tags(
            {
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "run_type": ctx.metadata.run_type.value,
                "model": ctx.metadata.model,
            }
        )

    def before_fit(self, dataset: Dataset, estimator: "Estimator") -> None:
        mlflow.set_tag("task", dataset.metadata.task.type.value)

    def after_test(self, dataset: Dataset, estimator: "Estimator", metrics: t.Dict) -> None:
        mlflow.log_metrics({name: float(metrics[name]) for name in METRICS[dataset.metadata.task.type]})


class TrainCallback(Callback):
    """keep_dataset, keep_model, keep_metrics, log_..."""

    def __init__(
        self,
        work_dir: t.Union[str, Path],
        hparams: t.Dict,
        ctx: Context,
        run_info_file: t.Optional[str] = None,
        data_info_file: t.Optional[str] = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.hparams = encode(copy.deepcopy(hparams))
        self.context: t.Dict = encode(ctx.metadata.to_json())
        self.run_info_file = run_info_file or "run_info.yaml"
        self.data_info_file = data_info_file or "data_info.yaml"

    def before_fit(self, dataset: Dataset, estimator: "Estimator") -> None:
        IO.save_yaml(dataset.metadata.to_json(), (self.work_dir / self.data_info_file).as_posix())
        IO.save_yaml(
            data={
                "estimator": {"cls": estimator.__class__.__name__, "params": encode(estimator.params)},
                "hparams": self.hparams,
                "context": self.context,
                "env": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None)},
            },
            file_path=self.work_dir / self.run_info_file,
            raise_on_error=False,
        )

    def after_fit(self, dataset: Dataset, estimator: "Estimator") -> None:
        estimator.save_model(self.work_dir)

    def after_test(self, dataset: Dataset, estimator: "Estimator", metrics: t.Dict) -> None:
        IO.save_yaml(encode(metrics), self.work_dir / "test_info.yaml")


class Estimator(object):
    def __init__(self, *args, **kwargs) -> None:
        self.params: t.Optional[t.Dict] = None
        self.model = None

    @abc.abstractmethod
    def save_model(self, save_dir: Path) -> None: ...

    @abc.abstractmethod
    def fit_model(self, dataset: Dataset, **kwargs) -> None: ...

    @classmethod
    def fit(cls, hparams: t.Dict, ctx: Context) -> t.Dict:
        """Train a model using hyperparameters in `params` and additional run details in `context`.

        context: callback (Callback), dataset (Dataset), problem (str), model (str), run_type (str), fit_params (dict),
        no_defaults
        """
        if ctx.dataset is None:
            ctx.dataset = build_dataset(ctx.metadata.dataset)

        if ctx.callbacks:
            callback: Callback = ContainerCallback(ctx.callbacks)
        else:
            callback = TrainCallback(IO.work_dir(), hparams, ctx)
            if mlflow.active_run() is not None:
                # When running with ray tune, there will be no active run.
                callback = ContainerCallback([callback, MLflowCallback(hparams, ctx)])

        if isinstance(callback, ContainerCallback):
            print("Estimator callbacks =", [c.__class__.__name__ for c in callback.callbacks])
        else:
            print("Estimator callbacks =", callback.__class__.__name__)

        estimator = cls(hparams, ctx.dataset.metadata)

        callback.before_fit(ctx.dataset, estimator)
        estimator.fit_model(ctx.dataset, **ctx.metadata.fit_params)
        callback.after_fit(ctx.dataset, estimator)

        metrics = estimator.evaluate(ctx.dataset, ctx)
        callback.after_test(ctx.dataset, estimator, metrics)

        return metrics

    def evaluate(self, dataset: Dataset, ctx: t.Optional[Context] = None, report: bool = True) -> t.Dict:
        if dataset.metadata.task.type.classification():
            metrics = self._evaluate_classifier(dataset)
        elif dataset.metadata.task.type.regression():
            metrics = self._evaluate_regressor(dataset)
        else:
            raise ValueError(f"Unsupported machine learning task {dataset.metadata.task}")
        """
        if report:
            if context.get('run_type', RunType.TRAIN) == RunType.HPO_WITH_RAY_TUNE:
                tune.report(**metrics)
            else:
                _metrics = ', '.join([f'{k}={metrics[k]}' for k in sorted(metrics.keys())])
                print(f"Model name={context['model']}, metrics={_metrics}")
        """
        return metrics

    def _evaluate_classifier(self, dataset: Dataset) -> t.Dict:
        """Report results of a training run.
        TODO: I can already have here results for train/valid (eval) splits.
        """
        metrics = {"dataset_accuracy": 0.0, "dataset_loss_total": 0.0, "dataset_loss_mean": 0.0}
        _num_examples = 0

        def _evaluate(x, y, name: str) -> None:
            nonlocal _num_examples
            predicted_probas = self.model.predict_proba(x)  # (n_samples, 2)
            predicted_labels = np.argmax(predicted_probas, axis=1)  # (n_samples,)
            metrics[f"{name}_accuracy"] = float(accuracy_score(y, predicted_labels))
            metrics[f"{name}_loss_mean"] = float(log_loss(y, predicted_probas, normalize=True))
            metrics[f"{name}_loss_total"] = float(metrics["train_loss_mean"] * len(y))

            if dataset.metadata.task.num_classes == 2:
                metrics[f"{name}_auc"] = float(roc_auc_score(y, predicted_probas[:, 1]))  # clas-1 probabilities
                metrics[f"{name}_f1"] = float(f1_score(y, predicted_labels))
                metrics[f"{name}_precision"] = float(precision_score(y, predicted_labels))
                metrics[f"{name}_recall"] = float(recall_score(y, predicted_labels))

            _num_examples += len(y)
            metrics["dataset_loss_total"] += metrics[f"{name}_loss_total"]
            metrics["dataset_accuracy"] += metrics[f"{name}_accuracy"] * len(y)

        for split_name in (DatasetSplit.TRAIN, DatasetSplit.VALID, DatasetSplit.TEST):
            split = dataset.split(split_name)
            if split is not None:
                _evaluate(split.x, split.y, split_name)

        if _num_examples > 0:
            metrics["dataset_accuracy"] = metrics["dataset_accuracy"] / _num_examples
            metrics["dataset_loss_mean"] = metrics["dataset_loss_total"] / _num_examples

        if DatasetSplit.VALID not in dataset.splits and DatasetSplit.TEST in dataset.splits:
            metrics.update(
                valid_loss_total=metrics["test_loss_total"],
                valid_loss_mean=metrics["test_loss_mean"],
                valid_accuracy=metrics["test_accuracy"],
            )

        return metrics

    def _evaluate_regressor(self, dataset: Dataset) -> t.Dict:
        metrics = {"dataset_mse": 0.0}
        _num_examples = 0

        def _evaluate(x, y, name: str) -> None:
            nonlocal _num_examples
            mse = mean_squared_error(y_true=y, y_pred=self.model.predict(x))
            metrics[f"{name}_mse"] = mse
            _num_examples += len(y)
            metrics["dataset_mse"] += mse * len(y)

        for split_name in (DatasetSplit.TRAIN, DatasetSplit.VALID, DatasetSplit.TEST):
            split = dataset.split(split_name)
            if split is not None:
                _evaluate(split.x, split.y, split_name)

        if _num_examples > 0:
            metrics["dataset_mse"] = metrics["dataset_mse"] / _num_examples

        if DatasetSplit.VALID not in dataset.splits and DatasetSplit.TEST in dataset.splits:
            metrics["valid_mse"] = metrics["test_mse"]

        return metrics

    @staticmethod
    def make_model(dataset_metadata: DatasetMetadata, classifier_cls, regressor_cls, params: t.Dict) -> t.Any:
        return classifier_cls(**params) if dataset_metadata.task.type.classification() else regressor_cls(**params)


_registry = ClassRegistry(base_cls="xtime.estimators.Estimator", path=Path(__file__).parent, module="xtime.estimators")


def get_estimator_registry() -> ClassRegistry:
    return _registry


def get_estimator(name: str) -> t.Type[Estimator]:
    return _registry.get(name)


def unit_test_train_model(test_case: TestCase, model_name: str, model_class: t.Any, ds: Dataset) -> t.Dict:
    model: Estimator = _registry.get(model_name)
    test_case.assertIs(
        model, model_class, f"model_name={model_name}, expected_model_class={model_class}, actual_model_class={model}"
    )
    metrics: t.Dict = model.fit(
        hparams=dict(n_estimators=1),
        ctx=Context(
            metadata=Metadata(dataset=ds.metadata.name, model=model_name, run_type=RunType.TRAIN),
            dataset=ds,
            callbacks=None,
        ),
    )
    test_case.assertIsInstance(metrics, dict)
    return metrics


def unit_test_check_metrics(test_case: TestCase, task: Task, metrics: t.Dict) -> None:
    if task.type.classification():
        expected_metrics = []
        for split in ("dataset", "train", "valid", "test"):
            expected_metrics.extend([f"{split}_accuracy", f"{split}_loss_mean", f"{split}_loss_total"])
    else:
        expected_metrics = ["dataset_mse", "train_mse", "valid_mse", "test_mse"]
    for metric in expected_metrics:
        test_case.assertIn(metric, metrics)
        test_case.assertIsInstance(metrics[metric], float)
        test_case.assertTrue(metrics[metric] >= 0)
