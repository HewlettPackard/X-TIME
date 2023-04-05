import json
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
import cuml

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
    
def _get_best_run_info(uri: str, use_numerical: bool = True) -> t.Dict:
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
    accuracy = 'test_mse' if dataset_info['Dataset']['task'] == 'REGRESSION' else 'test_accuracy'
    
    # Get the best Ray Tune trial that minimizes given metric
    best_trial: Trial = experiment.get_best_trial(perf_metric, mode='min')
    # Create return object
    model = mlflow_run.data.tags['model']
    models = dict(
        xgboost='model.ubj', light_gbm_clf='model.txt', catboost='model.bin', rf_clf='model.pkl', rf='model.pkl'
    )
    
    succeeded_trials: pd.DataFrame = experiment.results_df[experiment.results_df[perf_metric].notna()]
    
    if use_numerical == True:    
        X_test_path = "/opt/mlflow/datasets/numerical/"
    elif model in ["rf_clf"]:
        X_test_path = "/opt/mlflow/datasets/rf/"
    elif model in ["catboost", "rf"] or dataset_info['Dataset']['task'] == "REGRESSION":
        X_test_path = "/opt/mlflow/datasets/numerical/"
    else:
        X_test_path = "/opt/mlflow/datasets/gb/"
            
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
        'num_classes': 0 if dataset_info['Dataset']['task'] == 'REGRESSION' else dataset_info['Dataset']['num_classes'],
        # Test dataset path
        'X_test_path': X_test_path + dataset_info['Dataset']['name'].replace(":", "_") + "/test.pkl",
        # Dataset info
        'dataset_info': dataset_info,
        # Task
        'task': dataset_info['Dataset']['task']
    }
    return best_run_info

def load_dataset(model_info: t.Dict) -> t.Union[pd.DataFrame, xgboost.DMatrix]:
    with open(model_info['X_test_path'], "rb") as X_test_file:
        X_test = pickle.load(X_test_file)
        
    if model_info["model"] == "xgboost":
        return xgboost.DMatrix(X_test["x"])
    else:
        return X_test["x"]
    
def load_dataset_from_str(dataset_path: str, model_type: str, batch_size: int) -> t.Union[pd.DataFrame, xgboost.DMatrix]:
    with open(dataset_path, "rb") as X_test_file:
        X_test = pickle.load(X_test_file)
        
    X_test = X_test["x"]
    
    # With replacement
    selected = np.random.choice(X_test.shape[0], batch_size)
    
    X_test = X_test.iloc[selected, :]
    
    if model_type == "xgboost":
        return xgboost.DMatrix(X_test.apply(pd.to_numeric))
    else:
        return X_test

def load_model_from_str(model_path: str, model_type: str):
    if model_type == "xgboost":
        model = xgboost.Booster(model_file = model_path)
        model.set_param({'predictor': 'gpu_predictor'})
        
    elif model_type == "catboost":
        catboost_clf = catboost.CatBoostClassifier()
        model = catboost_clf.load_model(model_path)
    
    elif model_type == "light_gbm_clf":
        model = lightgbm.Booster(model_file = model_path)
        
    elif model_type in ["rf_clf", "rf"]:
        with open(model_path, "rb") as model_file:
            rf_model = pickle.load(model_file)
            
        model = cuml.ForestInference.load_from_sklearn(rf_model)
    else:
        raise ValueError(f"Unknown model type \"{model_type}\"")
        
    return model

def load_model(model_info: t.Dict) -> t.Dict:
    print(f"Loading model of type {model_info['model']}, dataset {model_info['dataset']}")
    
    if model_info["model"] == "xgboost":
        model = xgboost.Booster(model_file = model_info["model_path"])
        tree_df = model.trees_to_dataframe()
        
        model_info["n_trees"] = np.max(tree_df["Tree"]) + 1
        model_info["max_leaves"] = np.max(tree_df[tree_df["Feature"] == "Leaf"].groupby("Tree")["Feature"].count())
        
    elif model_info["model"] == "catboost":
        if model_info["task"] == "REGRESSION":
            catboost_clf = catboost.CatBoostClassifier()
        else:
            catboost_clf = catboost.CatBoostRegressor()
            
        model = catboost_clf.load_model(model_info["model_path"])
        
        if model_info["num_classes"] != 0:
            model_info["n_trees"] = model.tree_count_ * model_info["num_classes"]
        else:
            model_info["n_trees"] = model.tree_count_
            
        model_info["max_leaves"] = np.max([len(model._get_tree_leaf_values(t)) for t in range(model.tree_count_)])
    
    elif model_info["model"] == "light_gbm_clf":
        model = lightgbm.Booster(model_file = model_info["model_path"])
        tree_df = model.trees_to_dataframe()
        
        model_info["n_trees"] = np.max(tree_df["tree_index"])
        model_info["max_leaves"] = np.max(tree_df[tree_df.decision_type.isnull()].groupby("tree_index").node_depth.count())
        
    elif model_info["model"] in ["rf_clf", "rf"]:
        with open(model_info["model_path"], "rb") as model_file:
            model = pickle.load(model_file)
            
        if model_info["num_classes"] > 2:
            model_info["n_trees"] = model.n_estimators * model_info["num_classes"]
        else:
            model_info["n_trees"] = model.n_estimators
            
        model_info["max_leaves"] = np.max([estimator.get_n_leaves() for estimator in model.estimators_])
        
    else:
        raise ValueError(f"Unknown model type \"{model_info['model']}\"")
        
    model_info["trained_model"] = model
        
    return model_info