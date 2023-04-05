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

import numpy as np
import pandas as pd
import json
import os

import xgboost as xgb
import catboost
import sklearn.ensemble
import cuml

import pickle

from sklearn.model_selection import StratifiedKFold, train_test_split

from copy import deepcopy

default_model_filename = "tmp_model.json"
default_parameters_filename = "tmp_parameters.json"
default_log_file = "profile_results.csv"

def train_model_cb(X_data,
                   y_data,
                   parameters,
                   n_splits = 3,
                   model_filename = default_model_filename,
                   parameters_filename = default_parameters_filename):

    stratified_sampler = StratifiedKFold(n_splits = 3, shuffle = True)
    split_indices = [i for i in stratified_sampler.split(X_data, y_data)]

    X_train = X_data[split_indices[0][0]]
    y_train = y_data[split_indices[0][0]]

    X_test = X_data[split_indices[0][1]]
    y_test = y_data[split_indices[0][1]]

    model = catboost.CatBoostClassifier(**parameters).fit(X_data, y_data)

    model.save_model(model_filename, format = "cbm")

    with open(parameters_filename, "w") as file:
        json.dump(parameters, file)

    return model

def train_model_rf(X_data,
                   y_data,
                   parameters,
                   n_splits = 3,
                   model_filename = default_model_filename,
                   parameters_filename = default_parameters_filename):

    stratified_sampler = StratifiedKFold(n_splits = 3, shuffle = True)
    split_indices = [i for i in stratified_sampler.split(X_data, y_data)]

    X_train = X_data[split_indices[0][0]]
    y_train = y_data[split_indices[0][0]]

    X_test = X_data[split_indices[0][1]]
    y_test = y_data[split_indices[0][1]]

    model = sklearn.ensemble.RandomForestClassifier(**parameters).fit(X_data, y_data)

    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    with open(parameters_filename, "w") as file:
        json.dump(parameters, file)

    return model

def train_model_xgb(X_data,
                    y_data,
                    xgb_parameters,
                    n_splits = 3,
                    model_filename = default_model_filename,
                    parameters_filename = default_parameters_filename):

    stratified_sampler = StratifiedKFold(n_splits = 3, shuffle = True)
    split_indices = [i for i in stratified_sampler.split(X_data, y_data)]

    X_train = X_data[split_indices[0][0]]
    y_train = y_data[split_indices[0][0]]

    X_test = X_data[split_indices[0][1]]
    y_test = y_data[split_indices[0][1]]

    D_train = xgb.DMatrix(X_train, y_train)
    D_train.set_info()
    DX_test = xgb.DMatrix(X_test)

    parameters = deepcopy(xgb_parameters)
    del parameters["iterations"]

    booster = xgb.train(parameters, D_train, num_boost_round = xgb_parameters["iterations"])
    booster.save_model(model_filename)

    # Trying to use FIL directly
    # model = xgb.XGBClassifier(**parameters)
    # model.fit(X_train, y_train)
    # model.save_model(model_filename)

    with open(parameters_filename, "w") as file:
        json.dump(xgb_parameters, file)

    #return model

    return booster

def profile_inference(mlflow_info,
                      log_file = default_log_file,
                      tmp_log_file = "tmp.log",
                      partial_log_file = "partial_log.csv",
                      runs = 1,
                      n_splits = 3,
                      batch_size = 1,
                      unit = "ms",
                      keep_memcpy = False):

    # print(f"Running model: {mlflow_info['model']}")
    # print(f"With dataset: {mlflow_info['dataset']}")

    dataset_path = mlflow_info["X_test_path"]
    model_path = mlflow_info["model_path"]
    model_type = mlflow_info["model"]

    if keep_memcpy:
        keep_calls = ["memcpy"]
    else:
        keep_calls = []

    if model_type == "xgboost":
        keep_calls += ["PredictKernel"]
    elif model_type in ["rf_clf", "rf"]:
        keep_calls += ["infer_k"]
    elif model_type == "catboost":
        keep_calls += ["EvalObliviousTrees"]
    else:
        raise ValueError(f"Unknown model type \"{model_type}\"")

    os.environ["MODEL_PATH"] = model_path
    os.environ["DATASET_PATH"] = dataset_path
    os.environ["MODEL_TYPE"] = model_type
    os.environ["BATCH_SIZE"] = str(batch_size)

    inference_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow_inference.py")

    results_df = pd.DataFrame()

    for i in range(runs):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.system(f"nvprof --normalized-time-unit {unit} --csv --print-gpu-summary --log-file '{tmp_log_file}' python {inference_file}")
        os.system(f"tail -n +4 {tmp_log_file} > {partial_log_file}")
        #os.system(f"rm {tmp_log_file}")

        partial_df = pd.read_csv(partial_log_file)

        # Drop row with units
        partial_df.drop(0, inplace = True)
        partial_df.reset_index(drop = True, inplace = True)

        partial_df["run_id"] = i
        partial_df.rename(columns = {"Time": f"time_{unit}"}, inplace = True)
        partial_df["batch_size"] = batch_size

        partial_df = clean_profile(partial_df, keep = keep_calls)
        partial_df["time_proportion"] = partial_df["Time(%)"].astype(float) / 100.0

        partial_df = partial_df.drop(columns = ["Time(%)", "Type", "Calls",
                                                "Avg", "Min", "Max"])

        partial_df.rename(columns = {"Name": "kernel"}, inplace = True)

        if results_df.empty:
            results_df = partial_df
        else:
            results_df = pd.concat([results_df, partial_df])
            results_df.reset_index(drop = True, inplace = True)

        results_df.to_csv(log_file)

    #os.system(f"rm {partial_log_file}")

    #return results_df

    results_df["model"] = mlflow_info["model"]
    results_df["dataset"] = mlflow_info["dataset"]

    return results_df

def clean_profile(df, keep = ["PredictKernel"]):
    row_filter = "|".join(keep)
    filter_df = df[df["Name"].str.contains(row_filter, na = False)]
    filter_df.reset_index(inplace = True, drop = True)

    return filter_df
