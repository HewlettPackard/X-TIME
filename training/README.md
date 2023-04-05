# Prerequisites
This project was tested with several versions of python including `3.8.16`. Some or all of the available features
have been tested in Windows and Linux (Ubuntu) environments.
```shell
# Using `virtualenv`
virtualenv -p python3.9 venv
source venv/bin/activate
pip install -r requirements.txt

# Using conda
conda env create --name xtime-training --file environment.yml
conda activate xtime-training

# Optionally, install EDA and DEV dependencies
pip install -r ./requirements-eda.txt -r ./requirements-dev.txt
```

I have the following note in one of my Jupyter notebooks (do not recall the reason for this): 
> If on Windows OS, run the following cell. The 2nd command can fail - then go to your python directory and run it. 
> Once done, restart the kernel.
> ```shell
> !pip install pypiwin32
> !python Scripts\pywin32_postinstall.py -install
> ```

# Environment
There are no special requirements except users may need to provide information about their proxy servers. The `xtime`
may fail to download some or all datasets if proxy servers are not configured properly. To configure proxy servers,
export two environment variables `HTTP_PROXY` and `HTTPS_PROXY`.


# Datasets
For a list of datasets, Jupyter notebooks and preprocesssing pipelines see this [README](./xtime/datasets/README.md) 
file. Jupyter notebooks with exploratory data analysis (EDA) are available in the [notebooks](./notebooks/datasets) 
directory. 


# Machine Learning models
The [xtime.estimators](./xtime/estimators) package contains a number of machine learning models, including `CatBoost`,
`LightGBM`, `XGBoost` and some models from the Scikit-Learn ML library. The `xtime` uses Ray Tune for running 
hyperparameter search experiments.

As of 2023.03.28, the following models are present (`python -m xtime.main models list`):
```
- catboost
- lightgbm
- dummy
- rf
- xgboost
```

# Hyperparameters
Hyperparameters (HPs) are specified with the command line arguments `--params` option (as opposed to datasets and 
models that are positional arguments). By default, no hyperparameters are used. This means, that ML models will use
default values specified in their frameworks (such as `xgboost`). This also means that HPs search requires that users
provide HPs on a command line. There can be multiple sources of hyperparameters, and multiple sources can be used (in
this case HPs are loaded and merged with previously loaded HPs in the order they are specified on a command line).

In the context of this project, the following terms are used in teh context of hyperparameters:
- Hyperparameters (`HParams`): generic term that refers to HP values, search spaces or specifications (see below).
- Hyperparameter specification (`HParamsSpec`). This specification defines hyperparameter default value, type and 
  distribution to sample from that is considered as prior distribution for this parameter.
- Hyperparameter space (`HParamsSpace`) A hyperparameter search space. See the Ray Tune introduction to 
  [search spaces](https://docs.ray.io/en/latest/tune/api/search_space.html).

In the source code (and to some extent on a command line (see below)) collection of hyperparameters is always 
represented as a dictionary. Keys are names of hyperparameters. Values, depending on representation, could be values,
value specs of prior value distributions. 

The following is supported:
- `--params=default` Load default HPs for a given model. Default HPs for ML models have been borrowed from respective 
  papers. Default hyperparameters only exists in a context defined by, at least, a model (`xgboost`, `catboost` etc) and 
  a task (binary or multi-class classification, regression). 
- `--params=mlflow:///MLFLOW_RUN_ID` Load HPs from MLflow RUN. If this run is a hyperparameter search run, 
  hyperparameters associated with the best run are retrieved.
- `--params=params:lr=0.03;batch_size=128` Load hyperparameter from a provided string. This is a convenient way to 
  specify HPs on a command line. The string is parsed as a dictionary. Keys and values are separated by `=`. 
  Key-value pairs are separated by `;`. Values could include functions in `math` package, can use search spaces from 
  ray `tune` library (see [search spaces](https://docs.ray.io/en/latest/tune/api/search_space.html)), and also can use
  the `ValueSpec` class to define hyperparameter specification on a command line. When value is a string value, quote 
  and escape its value, e.g. `--params=params:tree_method=\"hist\"`.
  > The implementation is not secure and uses pythons `eval` function to parse parameter values.
- `--params=file:///mnt/space/ml/hparams.json` Load hyperparameters from a `JSON` file. The file should contain a 
  dictionary with hyperparameters. Other file types are supported (`YAML`, `YML`). The `file` scheme is optional.

The following examples demonstrate how to specify hyperparameters on a command line:
```shell
# Provide hyperparameters as a single argument
python -m xtime.main hparams query --params='params:lr=0.01;batch=tune.uniform(1, 128);n_estimators=ValueSpec(int, 100, tune.randint(100, 4001))'

# Provide hyperparameters as multiple arguments. Result is the same as above.
python -m xtime.main hparams query --params='params:lr=0.01;batch=tune.uniform(1, 128)' --params='params:n_estimators=ValueSpec(int, 100, tune.randint(100, 4001))'

# As it was mentioned above, to provide default (aka recommended) hyperparameters, one needs to provide a context.
# When running HP search experiments or training runs, the context is provided by users via positional DATASET and MODEL
# command line arguments.
python -m xtime.main hparams query --params=default --ctx='model="xgboost";task="multi_class_classification";num_classes=5'
```


# Tracking results of experiments
XTIME uses MLflow to track results of ML training and hyperparameter optimization experiments (but not for data 
preprocessing stages - basically, preprocessing is done on the fly at data load phase). By default, MLflow uses file 
system-based store for tracking experiment artifacts and metadata. The path to this store is usually an `mlruns` 
subdirectory in the current working directory. The following is the example (for Linux, but can easily be adapted for
Windows) how to start a custom instance of MLflow tracking server:
```shell
# Maybe run in a separate screen session
screen -S mlflow_server
# This is where run artifacts (e.g., files) will be stored. SQLite database for storing runs metadata will be stored in 
# the same directory.
export ARTIFACT_STORE=/opt/mlflow

# Adjust host and port parameters as needed.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
mlflow server --backend-store-uri sqlite:///${ARTIFACT_STORE}/mlruns.db --default-artifact-root=file://${ARTIFACT_STORE}/mlruns --host=0.0.0.0 --port=10000
```

Then, before running experiments, export MLflow URI:
```shell
export MLFLOW_TRACKING_URI=http://127.0.0.1:10000
```


# Running experiments
All entry points are implemented in [xtime.main](./xtime/main.py) module. Before running any scripts, make sure to 
add the `${PROJECT_ROOT}/training` to your `PYTHONPATH`:
```shell
# Linux
export PYTHONPATH=${PROJECT_ROOT}/training

# Windows
set PYTHONPATH=${PROJECT_ROOT}/training
```
It is assumed in the following examples that all scripts are executed in `${PROJECT_ROOT}/training` directory.

Start with studying output of these commands:
```shell
# High-level commands
python -m xtime.main --help

# Xtime stages related to datasets.
python -m xtime.main dataset --help

# Xtime stages related to ML models.
python -m xtime.main models --help

# Xtime stages related to machine learning experiments. 
python -m xtime.main experiment --help
```

The following example shows how to run a hyperparameter search experiment with 100 trials and random search.
```shell
# This dataset contains categorical features, so we need to "override" the default tree method for XGBoost. 
python -m xtime.main experiment search_hp telco_customer_churn:default xgboost random --params=default --params='params:tree_method="hist"' --num-search-trials=1 
```
Once the above command is executed, it will print out the MLflow run ID. This ID can be used to query MLflow for
hyperparameters associated with the best run. The following command will print out the best hyperparameters:
```shell
# Assuming that there's an MLflow run with ID 8037062a660847edabfede3ee3f6f4dc exists (hyperparameter search run).

# Get best hyperparameters as a dictionary
python -m xtime.main hparams query --params=mlflow:///8037062a660847edabfede3ee3f6f4dc

# Describe RUN (provide more detailed information)
python -m xtime.main experiment describe summary --run=mlflow:///8037062a660847edabfede3ee3f6f4dc
python -m xtime.main experiment describe best_trial --run=mlflow:///8037062a660847edabfede3ee3f6f4dc
```