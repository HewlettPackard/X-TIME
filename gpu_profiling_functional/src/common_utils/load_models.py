#
#  Copyright (2023) Hewlett Packard Enterprise Development LP
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
import sys

import mlflow
import ray
import yaml

import typing as t
import pandas as pd
import numpy as np

import lightgbm
import xgboost
import catboost
import sklearn.ensemble
import pickle

from multiprocessing import Process
from pathlib import Path

from mlflow.utils.file_utils import local_file_uri_to_path

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.trial import Trial

def _load_yaml(file_path: t.Union[str, Path]) -> t.Any:
    """Load YAML file.
    Args:
        file_path: Path to a YAML file.
    Returns:
        Content of this YAML file.
    """
    with open(file_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)

def get_best_run_info(uri: str) -> t.Dict:
    """
    Prerequisites:
        pip install mlflow ray[tune]
        export MLFLOW_TRACKING_URI=http://10.93.226.108:10000
        export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    Args:
        uri: MLflow run URI in the form `mlflow:///MLFLOW_RUN_ID` or just MLFLOW_RUN_ID
    Returns:
        Dictionary containing the following fields:
            - model_path: Local path to an ML model.
            - params_path: Local path to
            - model: Model name (catboost, xgboost)
    """
    # Extract MLflow run ID from `uri`
    mlflow_run_id = uri[10:] if uri.startswith('mlflow:///') else uri
    # Get MLflow run using run ID
    mlflow_run = mlflow.get_run(mlflow_run_id)
    # This run must be associated with hyperparameter search experiment
    if mlflow_run.data.tags['run_type'] != 'HPO_WITH_RAY_TUNE':
        raise ValueError(f"Unsupported MLflow run ({mlflow_run.data.tags['run_type']})")
    # Path where all files associated with this run are stored
    artifact_path: Path = Path(local_file_uri_to_path(mlflow_run.info.artifact_uri))
    # Ray Tune stores all its files in `ray_tune` subdirectory. Parse and get ray tune run summary,
    experiment = tune.ExperimentAnalysis((artifact_path / 'ray_tune').as_posix())
    # We need to retrieve the task for this run in order to identify the target metric.
    dataset_info = _load_yaml(artifact_path / 'dataset_info.yaml')
    perf_metric = 'valid_mse' if dataset_info['Dataset']['task'] == 'REGRESSION' else 'valid_loss_mean'
    accuracy = 'dataset_accuracy' #'valid_mse' if dataset_info['Dataset']['task'] == 'REGRESSION' else 'valid_loss_mean'
    # Get the best Ray Tune trial that minimizes given metric
    best_trial: Trial = experiment.get_best_trial(perf_metric, mode='min')
    # Create return object
    model = mlflow_run.data.tags['model']
    models = dict(
        xgboost='model.ubj', light_gbm_clf='model.txt', catboost='model.bin', rf_clf='model.pkl'
    )

    succeeded_trials: pd.DataFrame = experiment.results_df[experiment.results_df[perf_metric].notna()]

    best_run_info = {
        # Local path to ray tune trial directory
        'trial_path': best_trial.logdir,
        'params_path': (Path(best_trial.logdir) / 'params.json').as_posix(),
        # Local path to an ML model.
        'model_path': (Path(best_trial.logdir) / models[model]).as_posix(),
        # Model name (xgboost, light_gbm_clf, catboost, rf_clf)
        'model': model,
        # Target metric used by model
        'perf_metric': perf_metric,
        # Dataset name used in training/testing
        'dataset': dataset_info['Dataset']['name'],
        # Performance metric mean
        'perf_metric_mean': succeeded_trials[perf_metric].mean(),
        # Performance metric standard deviation
        'perf_metric_sd': succeeded_trials[perf_metric].std(),
        # Performance metric mean
        'accuracy_mean': succeeded_trials[accuracy].mean(),
        # Performance metric standard deviation
        'accuracy_sd': succeeded_trials[accuracy].std(),
        # Classes on dataset
        'num_classes': dataset_info['Dataset']['num_classes']
    }
    return best_run_info, dataset_info

def load_model(model_info: t.Dict) -> t.Dict:
    print(f"Loading model of type {model_info['model']}, dataset {model_info['dataset']}")

    if model_info["model"] == "xgboost":
        model = xgboost.Booster(model_file = model_info["model_path"])
        tree_df = model.trees_to_dataframe()

        model_info["n_trees"] = np.max(tree_df["Tree"]) + 1
        model_info["max_leaves"] = np.max(tree_df[tree_df["Feature"] == "Leaf"].groupby("Tree")["Feature"].count())

    elif model_info["model"] == "catboost":
        catboost_clf = catboost.CatBoostClassifier()
        model = catboost_clf.load_model(model_info["model_path"])

        if model_info["num_classes"] > 2:
            model_info["n_trees"] = model.tree_count_ * model_info["num_classes"]
        else:
            model_info["n_trees"] = model.tree_count_ * model_info["num_classes"]

        model_info["max_leaves"] = np.max([len(model._get_tree_leaf_values(t)) for t in range(model.tree_count_)])

    elif model_info["model"] == "light_gbm_clf":
        model = lightgbm.Booster(model_file = model_info["model_path"])
        tree_df = model.trees_to_dataframe()

        model_info["n_trees"] = np.max(tree_df["tree_index"])
        model_info["max_leaves"] = np.max(tree_df[tree_df.decision_type.isnull()].groupby("tree_index").node_depth.count())

    elif model_info["model"] == "rf_clf":
        with open(model_info["model_path"], "rb") as model_file:
            model = pickle.load(model_file)

        if model_info["num_classes"] > 2:
            model_info["n_trees"] = model.n_estimators * model_info["num_classes"]
        else:
            model_info["n_trees"] = model.n_estimators

        model_info["max_leaves"] = np.max([estimator.get_n_leaves() for estimator in model.estimators_])

    else:
        raise ValueError(f"Unknown model type \"{model_info['model']}\"")

    return model
