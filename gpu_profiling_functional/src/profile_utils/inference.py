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
import json
import numpy as np

import xgboost as xgb
import catboost
import sklearn.ensemble
import cuml

import pickle

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.datasets import fetch_covtype

from sklearn.metrics import accuracy_score

from mlflow_loader import load_dataset

n_splits = int(os.environ["N_SPLITS"])
batch_size = int(os.environ["BATCH_SIZE"])
dataset = os.environ["DATASET_FILENAME"]
model_filename = os.environ["MODEL_FILENAME"]
parameters_filename = os.environ["PARAMETERS_FILENAME"]
model_type = os.environ["MODEL_TYPE"]

if dataset == "covtype":
    X_data, y_data = fetch_covtype(return_X_y = True)
    y_data -= 1
else:
    raise ValueError(f"Dataset {dataset} unsuported")

stratified_sampler = StratifiedKFold(n_splits = n_splits, shuffle = True)
split_indices = [i for i in stratified_sampler.split(X_data, y_data)]

if batch_size > 0:
    chosen = np.random.choice(split_indices[0][1],
                              replace = False,
                              size = batch_size)
    X_test = X_data[chosen]
    y_test = y_data[chosen]
else:
    X_test = X_data[split_indices[0][1]]
    y_test = y_data[split_indices[0][1]]

if model_type == "xgboost":
    try:
        with open(parameters_filename, "r") as file:
            xgb_parameters = json.load(file)
            del xgb_parameters["iterations"]
    except:
        raise ValueError(f"Error parsing {parameters_filename}")

    try:
        booster = xgb.Booster(xgb_parameters)
        booster.load_model(model_filename)

        # Trying to use FIL directly
        # fil_model = cuml.ForestInference.load(model_filename, output_class = True)
    except:
        raise ValueError(f"Error parsing {model_filename}")

    DX_test = xgb.DMatrix(X_test)
    y_pred = booster.predict(DX_test)

    # Trying to use FIL directly
    # y_pred = fil_model.predict(X_test)

elif model_type == "rf":
    try:
        with open(model_filename, "rb") as model_file:
            model = pickle.load(model_file)
            fil_model = cuml.ForestInference.load_from_sklearn(model)
    except:
        raise ValueError(f"Error parsing {model_filename}")

    y_pred = fil_model.predict(X_test)

elif model_type == "cb":
    try:
        model = catboost.CatBoostClassifier()
        model.load_model(model_filename, format = "cbm")
    except:
        raise ValueError(f"Error parsing {model_filename}")

    print(f"X_test shape: {X_test.shape}")
    y_pred = model.predict(X_test, task_type = "GPU")

else:
    raise ValueError(f"Unknown model type \"{model_type}\"")
