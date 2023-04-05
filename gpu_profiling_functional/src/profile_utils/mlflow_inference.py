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
import mlflow_loader

import catboost
import xgboost
import sklearn.ensemble
import cuml

dataset_path = os.environ["DATASET_PATH"]
model_path = os.environ["MODEL_PATH"]
model_type = os.environ["MODEL_TYPE"]
batch_size = int(os.environ["BATCH_SIZE"])

model = mlflow_loader.load_model_from_str(model_path, model_type)
X_test = mlflow_loader.load_dataset_from_str(dataset_path, model_type, batch_size)

if model_type == "catboost":
    model.predict(X_test, task_type = "GPU")
else:
    model.predict(X_test)
