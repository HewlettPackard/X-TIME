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
import logging
import os
import typing as t
from pathlib import Path
from unittest import TestCase

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from xtime.contrib.mlflow_ext import MLflow
from xtime.datasets import Dataset, DatasetMetadata, DatasetSplit
from xtime.io import IO, encode
from xtime.ml import Task
from xtime.registry import ClassRegistry
from xtime.run import Context, Metadata, RunType

__all__ = ["Estimator", "get_estimator_registry", "get_estimator", "unit_test_train_model", "unit_test_check_metrics"]

logger = logging.getLogger(__name__)


class Callback(object):
    def before_fit(self, dataset: Dataset, estimator: "Estimator") -> None: ...

    def after_fit(self, dataset: Dataset, estimator: "Estimator") -> None: ...

    def after_test(self, dataset: Dataset, estimator: "Estimator", metrics: t.Dict[str, t.Any]) -> None:
        """Called after the model's final evaluation.

        Args:
            dataset: Dataset used to train/test this the model.
            estimator: Model
            metrics: Dictionary of metrics returned by `Estimator.evaluate` method.
        """
        ...


class ContainerCallback(Callback):
    def __init__(self, callbacks: t.Optional[t.List[Callback]]) -> None:
        self.callbacks = callbacks or []

    def before_fit(self, dataset: Dataset, estimator: "Estimator") -> None:
        for callback in self.callbacks:
            callback.before_fit(dataset, estimator)

    def after_fit(self, dataset: Dataset, estimator: "Estimator") -> None:
        for callback in self.callbacks:
            callback.after_fit(dataset, estimator)

    def after_test(self, dataset: Dataset, estimator: "Estimator", metrics: t.Dict[str, t.Any]) -> None:
        for callback in self.callbacks:
            callback.after_test(dataset, estimator, metrics)


class MLflowCallback(Callback):
    def __init__(self, hparams: t.Dict, ctx: Context) -> None:
        mlflow.log_params(hparams)
        MLflow.set_tags(dataset=ctx.metadata.dataset, run_type=ctx.metadata.run_type, model=ctx.metadata.model)

    def before_fit(self, dataset: Dataset, estimator: "Estimator") -> None:
        MLflow.set_tags(task=dataset.metadata.task)

    def after_test(self, dataset: Dataset, estimator: "Estimator", metrics: t.Dict[str, t.Any]) -> None:
        # TODO: (sergey) why did I come up with this implementation originally (in other words, why not dumping all
        #       metrics)? Is it because the metric value for some reason could be non-numeric?
        # mlflow.log_metrics({name: float(metrics[name]) for name in METRICS[dataset.metadata.task.type]})

        loggable_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                loggable_metrics[name] = float(value)
        mlflow.log_metrics(loggable_metrics)


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

    def after_test(self, dataset: Dataset, estimator: "Estimator", metrics: t.Dict[str, t.Any]) -> None:
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
            ctx.dataset = Dataset.create(ctx.metadata.dataset)

        if ctx.callbacks:
            callback: Callback = ContainerCallback(ctx.callbacks)
        else:
            callback = TrainCallback(IO.work_dir(), hparams, ctx)
            if mlflow.active_run() is not None:
                callback = ContainerCallback([callback, MLflowCallback(hparams, ctx)])

        estimator = cls(hparams, ctx.dataset.metadata)

        callback.before_fit(ctx.dataset, estimator)
        estimator.fit_model(ctx.dataset, **ctx.metadata.fit_params)
        callback.after_fit(ctx.dataset, estimator)

        metrics: t.Dict = estimator.evaluate(ctx.dataset)
        callback.after_test(ctx.dataset, estimator, metrics)

        return metrics

    def evaluate(self, dataset: Dataset) -> t.Dict[str, t.Any]:
        if dataset.metadata.task.type.classification():
            metrics = self._evaluate_classifier(dataset)
        elif dataset.metadata.task.type.regression():
            metrics = self._evaluate_regressor(dataset)
        else:
            raise ValueError(f"Unsupported machine learning task {dataset.metadata.task}")
        return metrics

    def _evaluate_classifier(self, dataset: Dataset) -> t.Dict[str, t.Any]:
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

    def _evaluate_regressor(self, dataset: Dataset) -> t.Dict[str, t.Any]:
        """Evaluate regressor model."""
        # Baseline predictor (strategy=mean) will be used to compute out-of-sample R-squared metric.
        baseline_y_pred: t.Optional[float] = None
        train_y_vals: np.ndarray = dataset.splits[DatasetSplit.TRAIN].y.values
        if train_y_vals.ndim == 1 or (train_y_vals.ndim == 2 and train_y_vals.shape[1] == 1):
            baseline_y_pred = train_y_vals.flatten().mean()
        else:
            logger.debug("Out-of-sample R-squared will not be computed (shape=%s)", str(train_y_vals.shape))
        #
        metrics = {"dataset_mse": 0.0}
        _num_examples = 0

        def _evaluate(x: pd.DataFrame, y: t.Union[pd.DataFrame, pd.Series], name: str) -> None:
            nonlocal _num_examples
            _num_examples += len(y)

            y_pred = self.model.predict(x)

            mse_metric_name = f"{name}_mse"
            metrics[mse_metric_name] = mean_squared_error(y_true=y, y_pred=y_pred)
            metrics["dataset_mse"] += metrics[mse_metric_name] * len(y)

            if baseline_y_pred is not None:
                _benchmark_mse: float = float(np.average((y - baseline_y_pred) ** 2, axis=0))
                metrics[f"{name}_r2_oos"] = 1.0 - metrics[mse_metric_name] / _benchmark_mse

            metrics[f"{name}_r2"] = r2_score(y_true=y, y_pred=y_pred)

        for split_name in (DatasetSplit.TRAIN, DatasetSplit.VALID, DatasetSplit.TEST):
            split: t.Optional[DatasetSplit] = dataset.split(split_name)
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
