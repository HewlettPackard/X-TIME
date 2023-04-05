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