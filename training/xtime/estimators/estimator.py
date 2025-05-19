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
import importlib
import logging
import os
import typing as t
from dataclasses import dataclass
from pathlib import Path
from unittest import TestCase

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ray.air import session as ray_session
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from xtime.contrib.mlflow_ext import MLflow
from xtime.datasets import Dataset, DatasetMetadata, DatasetSplit
from xtime.errors import ignore_exceptions
from xtime.io import IO, encode
from xtime.ml import ClassificationTask, Task, TaskType
from xtime.registry import ClassRegistry
from xtime.run import Context, Metadata, RunType

__all__ = [
    "Estimator",
    "get_estimator_registry",
    "get_estimator",
    "LegacySavedModelInfo",
    "Model",
    "get_expected_available_estimators",
    "unit_test_train_model",
    "unit_test_check_metrics",
]

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
        model_info_file: t.Optional[str] = None,
        test_info_file: t.Optional[str] = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.hparams = encode(copy.deepcopy(hparams))
        self.context: t.Dict = encode(ctx.metadata.to_json())
        self.run_info_file = run_info_file or "run_info.yaml"
        self.data_info_file = data_info_file or "data_info.yaml"
        self.model_info_file = model_info_file or "model_info.yaml"
        self.test_info_file = test_info_file or "test_info.yaml"

    def before_fit(self, dataset: Dataset, estimator: "Estimator") -> None:
        IO.save_yaml(dataset.metadata.to_json(), (self.work_dir / self.data_info_file).as_posix())
        IO.save_yaml(
            data={
                "estimator": {"cls": estimator.__class__.__name__, "params": encode(estimator.params)},
                "hparams": self.hparams,
                "context": self.context,
                "env": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None)},
                "metadata": {
                    "run": self.run_info_file,
                    "data": self.data_info_file,
                    "model": self.model_info_file,
                    "test": self.test_info_file,
                },
            },
            file_path=self.work_dir / self.run_info_file,
            raise_on_error=False,
        )

    def after_fit(self, dataset: Dataset, estimator: "Estimator") -> None:
        estimator.save_model(self.work_dir)
        metadata = {
            "model": {
                "name": getattr(estimator, "NAME", None),
                "class": estimator.__class__.__name__,
                "type": dataset.metadata.task.type.value,
                "features": [feature.to_json() for feature in dataset.metadata.features],
            }
        }
        if isinstance(dataset.metadata.task, ClassificationTask):
            metadata["model"]["num_classes"] = dataset.metadata.task.num_classes
        IO.save_yaml(metadata, self.work_dir / self.model_info_file, raise_on_error=False)

    def after_test(self, dataset: Dataset, estimator: "Estimator", metrics: t.Dict[str, t.Any]) -> None:
        IO.save_yaml(encode(metrics), self.work_dir / self.test_info_file)


class Estimator:
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
        else:
            logger.info("Not loading dataset - it has already been loaded.")
            # TODO sergey Study how object store works with pandas data frames (zero-copy?). With python 3.12 where we
            #      upgraded version of many libraries (ray, scikit-learn, lightgbm and others) something does not work,
            #      and this seems to be related to the fact how ray makes these data frames available in its workers
            #      given that they are originally placed into the data store. This only manifests itself when
            #      `lightgbm` classifiers are used.
            if getattr(cls, "NAME", None) == "lightgbm":
                if hasattr(ray_session, "_get_session"):
                    get_session: t.Callable = getattr(ray_session, "_get_session")
                elif hasattr(ray_session, "get_session"):
                    get_session = getattr(ray_session, "_get_session")
                else:

                    def get_session() -> None:
                        return None

                if get_session() is not None:
                    # we are in Ray Session
                    logger.warning(
                        "In active ray session and model is LightGBM. The dataset's deep copy will be created. "
                        "This needs to be debugged - see the TODO comment in the source code."
                    )
                    ctx.dataset = copy.deepcopy(ctx.dataset)
        dataset: Dataset = ctx.dataset

        if ctx.callbacks:
            callback: Callback = ContainerCallback(ctx.callbacks)
        else:
            callback = TrainCallback(IO.work_dir(), hparams, ctx)
            if mlflow.active_run() is not None:
                callback = ContainerCallback([callback, MLflowCallback(hparams, ctx)])

        estimator = cls(hparams, dataset.metadata)

        callback.before_fit(dataset, estimator)
        estimator.fit_model(dataset, **ctx.metadata.fit_params)
        callback.after_fit(dataset, estimator)

        metrics: t.Dict = estimator.evaluate(dataset)
        callback.after_test(dataset, estimator, metrics)

        return metrics

    def evaluate(self, dataset: Dataset, **kwargs) -> t.Dict[str, t.Any]:
        if dataset.metadata.task.type.classification():
            metrics = self._evaluate_classifier(dataset, **kwargs)
        elif dataset.metadata.task.type.regression():
            if kwargs:
                logger.warning(
                    "The regressor evaluation method does not support any additional arguments (%s). "
                    "They will be ignored.",
                    kwargs,
                )
            metrics = self._evaluate_regressor(dataset)
        else:
            raise ValueError(f"Unsupported machine learning task {dataset.metadata.task}")
        return metrics

    def _evaluate_classifier(
        self, dataset: Dataset, predict_proba_kwargs: t.Optional[t.Dict] = None
    ) -> t.Dict[str, t.Any]:
        """Report results of a training run.

        TODO: I can already have here results for train/valid (eval) splits.

        Args:
            dataset: Dataset to evaluate this model on.
            predict_proba_kwargs: Key-value arguments for the `model.predict_proba` call. These arguments are model
                specific, and can be used, for instance, to limit number of trees in a tree-based models (e.g.,
                XGBoost).

        Returns:
            Dictionary with metrics computed on different splits for the input `datasets`.
        """
        predict_proba_kwargs = predict_proba_kwargs or {}

        metrics = {"dataset_accuracy": 0.0, "dataset_loss_total": 0.0, "dataset_loss_mean": 0.0}
        _num_examples = 0

        assert isinstance(
            dataset.metadata.task, ClassificationTask
        ), "Invalid task type (expecting `ClassificationTask` task)."
        task: ClassificationTask = dataset.metadata.task

        def _evaluate(x, y: t.Union[np.ndarray, pd.Series], name: str) -> None:
            nonlocal _num_examples

            if isinstance(y, pd.Series):
                y = y.values
            if not isinstance(y, np.ndarray):
                logger.warning(
                    "Expecting y (true labels) to be of type numpy ndarray, but actual type - '%s'.", type(y)
                )

            # (n_samples, 2)
            predicted_probas: t.Union[np.ndarray, pd.DataFrame] = self.model.predict_proba(x, **predict_proba_kwargs)
            if isinstance(predicted_probas, pd.DataFrame):
                # This is the case for (some?) models from RAPIDS library.
                predicted_probas = predicted_probas.values
            if not isinstance(predicted_probas, np.ndarray):
                logger.warning(
                    "Expecting 'model.predict_proba' to return np.ndarray type. Actual returned type - '%s'.",
                    type(predicted_probas),
                )
            predicted_labels = np.argmax(predicted_probas, axis=1)  # (n_samples,)
            metrics[f"{name}_accuracy"] = float(accuracy_score(y, predicted_labels))
            metrics[f"{name}_loss_mean"] = float(log_loss(y, predicted_probas, normalize=True))
            metrics[f"{name}_loss_total"] = float(metrics["train_loss_mean"] * len(y))

            if task.num_classes == 2:
                metrics[f"{name}_auc"] = float(roc_auc_score(y, predicted_probas[:, 1]))  # class-1 probabilities
                metrics[f"{name}_f1"] = float(f1_score(y, predicted_labels))
                metrics[f"{name}_precision"] = float(precision_score(y, predicted_labels))
                metrics[f"{name}_recall"] = float(recall_score(y, predicted_labels))
                compute_and_save_precision_recall_curve(name, y, predicted_probas)

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


@dataclass
class LegacySavedModelInfo:
    """Information about saved model that is used to load the model.

    This will be considered a legacy format once the XTIME project starts to use MLflow model logging feature.
    """

    model: str = ""
    """catboost,lightgbm,xgboost,dummy,rf,rf_clf"""
    task: str = ""
    """binary_classification,multi_class_classification,regression"""

    def __str__(self) -> str:
        return f"LegacySavedModelInfo(model={self.model}, task={self.task})"

    def is_valid(self) -> bool:
        """Return true if combination of model and task can be resolved to a model file name and class type."""
        return self.model in {
            "catboost",
            "lightgbm",
            "xgboost",
            "dummy",
            "rf",
            "rf_clf",
            "rapids-rf",
        } and self.task in {"binary_classification", "multi_class_classification", "regression"}

    def file_name(self) -> str:
        """Return model file name."""
        model_to_file_name: t.Dict[str, str] = {
            "catboost": "model.bin",
            "lightgbm": "model.txt",
            "xgboost": "model.ubj",
            "dummy": "model.pkl",
            "rf": "model.pkl",
            "rf_clf": "model.pkl",
            "rapids-rf": "model.pkl",
        }
        if self.model in model_to_file_name:
            return model_to_file_name[self.model]

        raise ValueError(f"Unsupported model type (model={self.model}).")

    @staticmethod
    def get_file_name(model: str) -> str:
        return LegacySavedModelInfo(model=model).file_name()

    @classmethod
    def from_path(cls, path: Path) -> t.Optional["LegacySavedModelInfo"]:
        """Determine saved model details using information from files created by xtime stages.

        Files that are read in this function are created by the TrainingCallback class.
        """
        if path.is_file():
            path = path.parent

        info = LegacySavedModelInfo()

        if (path / "model_info.yaml").is_file():
            # This is a new file that is not available for past experiments.
            model_info: t.Dict = IO.load_dict(path / "model_info.yaml")
            info.model = model_info.get("model", {}).get("name", "")
            info.task = model_info.get("model", {}).get("type", "")
        else:
            # Else, try standard run_info and data_info files.
            if (path / "run_info.yaml").is_file():
                run_info: t.Dict = IO.load_dict(path / "run_info.yaml")
                info.model = run_info.get("context", {}).get("model", "")

            if (path / "data_info.yaml").is_file():
                data_info: t.Dict = IO.load_dict(path / "data_info.yaml")
                info.task = data_info.get("task", {}).get("type", "")

        if info.is_valid():
            return info

        logger.warning(
            "Cannot identify model (path=%s) as legacy saved model (model='%s', task='%s').",
            path.as_posix(),
            info.model,
            info.task,
        )
        return None


class LegacySavedModelLoader:
    """Class that can load ML models in legacy format.

    Legacy format is the serialization format before XTIME starts to use MLflow model logging feature.
    """

    @staticmethod
    def load_model(path: Path, legacy_saved_model_info: LegacySavedModelInfo) -> t.Any:
        """Load a model from a given path.

        Args:
            path: Path to a directory or model file.
            legacy_saved_model_info: Information about the model to be loaded.

        Returns:
            ML model. The actual type depends on model name and task (could be XGBClassifier, XGBRegressor,
            CatBoostClassifier, CatBoostRegressor, lightgbm.Booster and models from scikit-learn library).
        """
        model_file: str = legacy_saved_model_info.file_name()
        if not (path / model_file).is_file():
            raise FileNotFoundError(f"No model file ('{model_file}') found in '{path}' for {legacy_saved_model_info}.")

        model_name = legacy_saved_model_info.model
        task_type = TaskType(legacy_saved_model_info.task)

        if model_name == "xgboost":
            import xgboost

            model = xgboost.XGBClassifier() if task_type.classification() else xgboost.XGBRegressor()
            model.load_model(path / model_file)
        elif model_name == "catboost":
            import catboost

            model = catboost.CatBoostClassifier() if task_type.classification() else catboost.CatBoostRegressor()
            model.load_model((path / model_file).as_posix())
        elif model_name == "lightgbm":
            import lightgbm

            model = lightgbm.Booster(model_file=(path / model_file).as_posix())
        elif model_name in {"dummy", "rf", "rf_clf"}:
            import pickle

            with open(path / model_file, "rb") as file:
                model = pickle.load(file)
        elif model_name == "rapids-rf":
            import pickle

            with open(path / model_file, "rb") as file:
                model = pickle.load(file)
        else:
            raise NotImplementedError(f"Model loading ({model_name}) has not been implemented yet.")

        return model


class Model:
    """Class to load models serialized by estimators.

    TODO sergey: get rid of this class, or replace it with something like TrainRun (needs to be always consistent
        with the TrainingCallback class that actually writes the data used by this class). The TrainRun can also server
        as a single entry point to all run metadata.
    """

    @staticmethod
    def get_file_name(model_name: str) -> str:
        """Return model file name based upon model name.

        Args:
            model_name: Name of a model (xgboost, catboost, etc.). See `Model._model_files` dictionary.

        Returns:
            A file name for a given model.
        """
        return LegacySavedModelInfo.get_file_name(model_name)

    @staticmethod
    def load_model(path: Path, legacy_saved_model_info: t.Optional[LegacySavedModelInfo] = None) -> t.Any:
        """Load model stored in a given path.

        Args:
            path: Directory path where a model is stored (e.g., MLflow artifact path or Ray Tune trial directory).
            legacy_saved_model_info: When model is a legacy saved model, this info provides information about this
                model. If it's None, the function will try to figure out this information from the path.

        Returns:
            Instance of a model. This will be an instance of a model of a respective framework (e.g.,
                xgboost.XGBRegressor, catboost.CatBoostClassifier, etc.).
        """
        if legacy_saved_model_info is not None and not legacy_saved_model_info.is_valid():
            raise ValueError(f"Provided legacy saved model info is not valid ({legacy_saved_model_info}).")

        if legacy_saved_model_info is not None:
            return LegacySavedModelLoader.load_model(path, legacy_saved_model_info)

        legacy_saved_model_info = LegacySavedModelInfo.from_path(path)
        if legacy_saved_model_info is not None:
            return LegacySavedModelLoader.load_model(path, legacy_saved_model_info)

        raise ValueError(f"Cannot load model from '{path}'.")


def get_expected_available_estimators() -> t.List[str]:
    """Return list of estimators in sorted order that are expected to be available in this system."""
    # Mapping from a library name to an estimator name.
    libraries = {"catboost": ["catboost"], "lightgbm": ["lightgbm"], "cuml": ["rapids-rf"], "xgboost": ["xgboost"]}
    estimators = ["dummy", "rf"]  # These must always be available (scikit-learn - mandatory dependency).
    for lib_name, lib_estimators in libraries.items():
        try:
            _ = importlib.import_module(lib_name)
            estimators.extend(lib_estimators)
        except ImportError:
            ...
    return sorted(estimators)


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


@ignore_exceptions()
def compute_and_save_precision_recall_curve(
    split_name: str, true_labels: np.ndarray, predicted_scores: np.ndarray
) -> None:
    """Compute and save (csv and png) precision-recall curve.

    Args:
        true_labels (np.ndarray): Array of true labels.
        predicted_scores (np.ndarray): Array of predicted scores (shape - [num_examples, num_classes]).
    """
    # Compute precision, recall and thresholds
    precision, recall, thresholds = precision_recall_curve(true_labels, predicted_scores[:, 1])
    base_file_name: str = f"{split_name}_precision_recall"

    # Save as YAML file.
    IO.save_yaml(
        {"precision": precision.tolist(), "recall": recall.tolist(), "threshold": thresholds.tolist()},
        IO.work_dir() / f"{base_file_name}.yaml",
    )

    # Save as PNG file.
    fig, precision_ax = plt.subplots()
    major_ticks = np.arange(0.1, 1.1, 0.1).tolist()
    minor_ticks = np.arange(0.05, 1.05, 0.1).tolist()

    # In newer versions this will update axes parameters (xlabel, xlim, ylim and maybe others) that results in different
    # output charts on different systems. To avoid correcting it, this is commented.
    # from sklearn.metrics import PrecisionRecallDisplay
    # viz = PrecisionRecallDisplay(precision=precision, recall=recall)
    # _ = viz.plot(ax=precision_ax)

    # Instead, it is done here which is ~ consistent with PrecisionRecallDisplay.plot.
    (pr_curve,) = precision_ax.plot(recall, precision, drawstyle="steps-post", color="b")
    precision_ax.set(xlim=[0.00, 1.05], ylim=[0.00, 1.05], xlabel="Recall", xticks=major_ticks, yticks=major_ticks)
    precision_ax.set_ylabel("Precision", color="b")
    precision_ax.set_xticks(minor_ticks, minor=True)
    precision_ax.set_yticks(minor_ticks, minor=True)
    precision_ax.grid(visible=True, which="major", linestyle="-", axis="both")
    precision_ax.grid(visible=True, which="minor", linestyle="--", axis="both")

    thresholds_ax = precision_ax.twinx()
    pt_curve = thresholds_ax.scatter(recall[0:-1], thresholds, label="threhsold", c="g", s=1)
    thresholds_ax.set(yticks=major_ticks, ylim=[0.00, 1.05])
    thresholds_ax.set_yticks(minor_ticks, minor=True)
    thresholds_ax.set_ylabel("Threshold", color="g")

    precision_ax.legend([pr_curve, pt_curve], ["precision", "threshold"])

    fig.savefig(IO.work_dir() / f"{base_file_name}.png", bbox_inches="tight")
    plt.close(fig)
