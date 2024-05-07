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

import copy
import json
import os
import typing as t
from pathlib import Path

import numpy as np
from ray.tune.search.sample import Domain
from ray.tune.search.variant_generator import generate_variants

from xtime.io import IO, PathLike
from xtime.run import RunType

__all__ = [
    "HParamsSource",
    "HParams",
    "HParamsSpace",
    "JsonEncoder",
    "ValueSpec",
    "HParamsSpec",
    "get_hparams",
    "from_auto",
    "from_mlflow",
    "from_string",
    "from_file",
]


HParamsSource = t.Union[t.Dict, str, t.Iterable[t.Union[str, t.Dict]]]
"""Specification options for hyperparameters (HP).

    - dict: Ready-to-use dictionary of HPs mapping HP names to values. 
    - tuple, list: Sequence of specifications (e.g., multiple sources). Merging is done left to right, e.g.,
        sequence [hparams1, hparams2, hparams3] results in 
        `hps = hparams1.copy(); hps.update(hparams2); hps.update(hparams3)`.
    - str: string representation of HPs or URI of where HPs should be retrieved from. Options are:
        - `mlflow:///MLFLOW_RUN_ID` retrieve HPs from MLflow run (train or HP search).
        - `params:lr=0.3;batch=128` in-place specification.
        - `file:params.yaml` Load HPs from one of supported file formats.
        - ''
"""

HParams = t.Dict
"""Dictionary of hyperparameters. It should map from a parameter name to a parameter (primitive) value."""

HParamsSpace = t.Dict
"""Hyperparameter search space (maybe, framework-specific). But can also map a parameter name to just its value."""


class JsonEncoder(json.JSONEncoder):
    """Custom JSON encoder to support additional types."""

    def default(self, obj):
        if isinstance(obj, ValueSpec):
            return {"dtype": str(obj.dtype), "default": obj.default, "space": self.default(obj.space)}
        if isinstance(obj, Domain):
            return {"domain": obj.domain_str, "sampler": str(obj.sampler)}
        return super().default(obj)


class ValueSpec(object):
    """Hyperparameter value specification defining its type, default value and prior distribution to sample from."""

    def __init__(self, dtype: t.Callable, default: t.Any, space: t.Any) -> None:
        """Initialize this instance.

        Args:
            dtype: Callable  object (e.g., python types - int, float etc.) that returns/converts values into appropriate
                type. Is used to make sure that ML models get the right types.
            default: Default value for this hyperparameter. Semantics of this to be defined. One possible option could
                be to use Nones to allow underlying ML models to select the appropriate defaults.
            space: Prior distribution for this parameter to sample from.
        """
        self.dtype = dtype
        self.default = default
        self.space = space


class HParamsSpec(object):
    """Helper class to define hyperparameters that provide default value, search space and random sample.

    Args:
        kwargs: dictionary mapping HP name to its value specification (ValueSpec).
    """

    def __init__(self, **kwargs) -> None:
        for name, value in kwargs.items():
            assert isinstance(value, ValueSpec), f"Invalid HP value spec ({value}) for HP `{name}`."
        self.params: t.Dict[str, ValueSpec] = copy.deepcopy(kwargs)

    def default(self) -> HParams:
        """Return dictionary with default values."""
        return {name: value.default for name, value in self.params.items()}

    def space(self) -> HParamsSpace:
        """Return search space for this set of hyperparameters."""
        return {name: value.space for name, value in self.params.items()}

    def sample(self) -> t.Optional[HParams]:
        """Return a random sample from a search space."""
        configs = [config for _, config in generate_variants(self.space())]
        return np.random.choice(configs) if configs else None

    def merge(self, *args: t.Dict, use_default: bool = True) -> t.Dict:
        _params = self.default() if use_default else {}
        for arg in args:
            _params.update(arg or {})
        for name in _params.keys():
            if name in self.params and _params[name] is not None:
                _params[name] = self.params[name].dtype(_params[name])
        return _params


def get_hparams(source: t.Optional[HParamsSource] = None) -> t.Dict:
    """Retrieve/convert hyperparameters from/in source to standard representation - dictionary.
    Args:
        source: Source or sources of hyperparameters. For possible values and how they are processed see the source
            code of this function.
    """

    hp_dict: t.Any = None
    if source is None:
        # Just empty dict of hyperparameters.
        hp_dict = {}
    elif isinstance(source, dict):
        # It is assumed that source is already a valid hyperparameter dictionary.
        hp_dict = copy.deepcopy(source)
    elif isinstance(source, (list, tuple)):
        # This is a list of multiple sources.
        hp_dict = {}
        for one_source in source:
            hp_dict.update(get_hparams(one_source))
    elif isinstance(source, str):
        source = source.strip()
        if source.startswith("auto:"):
            hp_dict = from_auto(source)
        elif source.startswith("mlflow:///"):
            # Retrieve hyperparameters from an MLflow run.
            hp_dict = from_mlflow(source)
        elif source.startswith("file:") or os.path.isfile(source):
            # Retrieve hyperparameters from a file.
            hp_dict = from_file(source)
        else:
            # Try to parse string that contains hyperparameters.
            hp_dict = from_string(source)

    if isinstance(hp_dict, dict):
        return hp_dict

    raise ValueError(
        f"Unsupported source of hparams (source=(value={source}, type={type(source)}). Extracted hyperparameters "
        f"expected to be a dictionary but hp_dict=(value={hp_dict}, type={type(hp_dict)})."
    )


def from_auto(params: t.Optional[str] = None) -> t.Dict:
    """Request hyperparameters from an external entity such as hyperparameter recommender.

    Args:
        params: A string containing entity name and query parameters. It must have the following format:
            [auto:]NAME:QUERY. The `auto:` is an optional scheme specification that differentiates this HP specification
            from others such MLflow runs, files, etc. The NAME (string) is the name of the entity to retrieve
            hyperparameters from. The QUERY is generally an optional key-value semicolon-separated string that defines
            context for the problem for which hyperparameters should be retrieved. It's the same format used for
            hyperparameters (e.g., `lr=0.1;batch=128`). Example specifications:
                - `auto:default:model=xgboost;task=binary_classification` Retrieve hyperparameters from a hyperparameter
                    recommender model identified with the `default` name for XGBoost model and binary classification
                    problems.
    Returns:
         Dictionary with hyperparameter values or hyperparameters search spaces.

    Supported entities:
        - `default` Default recommender that's part of this project. Defines common hyperparameters and their search
            spaces for multiple models (XGBoost, LightGBM, CatBoost and others) amd multiple tasks (regression, binary
            and multi-class classification).

    """
    params = _str_content(params, "auto:")
    if not params:
        return {}

    source_and_query: t.List[str] = params.split(":", maxsplit=1)
    source, query_str = (source_and_query[0], None) if len(source_and_query) == 1 else source_and_query
    query_params: t.Dict = from_string(query_str)

    if source == "default":
        from tinydb import Query

        from xtime.hparams.recommender import DefaultRecommender

        query = Query()
        model = query_params.pop("model", None)
        if model:
            query = query.tags.model == model
        task = query_params.pop("task", None)
        if task:
            query = query & Query().tags.tasks.any([task])
        run_type = query_params.pop("run_type", "train")
        if query_params:
            raise ValueError(
                "The implementation is in its early stages, and we only support the following parameters - `model`, "
                "`task` and `run_type`."
            )

        recommender = DefaultRecommender()
        if run_type == RunType.TRAIN.value:
            recommendations: t.List[t.Dict] = recommender.recommend_default_values(query)
        elif run_type == RunType.HPO.value:
            recommendations: t.List[t.Dict] = recommender.recommend_search_space(query)
        else:
            raise ValueError(f"Unknown run_type={run_type}. Must be one of 'train', 'hpo'.")

        if recommendations:
            return recommendations[0]
        return {}
    else:
        raise RuntimeError(f"Unsupported hyperparameter source (params={params}, source={source}).")


def from_string(params: t.Optional[str] = None) -> t.Dict:
    """Parse hyperparameters provided by users on a command line.

    Args:
        params: If it's a string, a semicolon-separated parameters in the form NAME=VALUE. If it's a list, each item is
            a string of the form NAME=VALUE. Names are strings, while VALUEs are parsed with `eval` function. They can
            use the following packages: math, tune and Value class.

    Examples:
        --params='depth=8'
        --params='learning_rate=0.01;search_method="beam";subsample=True'

    Returns:
        Dictionary of hyperparameters.
    """
    # Check if input is empty
    params = _str_content(params, "params:")
    if not params:
        return {}

    # These imports may be required by the `eval` call below.
    import math  # noqa # pylint: disable=unused-import

    from ray import tune  # noqa # pylint: disable=unused-import

    from xtime.hparams import ValueSpec  # noqa # pylint: disable=unused-import

    # Iterate over each parameter and parse it.
    hp_dict = {}
    for idx, param in enumerate(params.split(";")):
        # Check of this is an empty spec (e.g., `;` at the end such as "params:lr=0.3;batch=128;")
        param = param.strip()
        if not param:
            continue
        #
        try:
            name, value = param.split("=", maxsplit=1)
        except ValueError as err:
            raise ValueError(
                f"Invalid parameter in from_string (params={params}, idx={idx}, param={param}). Cannot split parameter "
                "spec (using '=' character) into parameter name and parameter value."
            ) from err

        name = name.strip()
        if not name.isidentifier():
            raise ValueError(
                f"Invalid parameter in from_string (params={params}, idx={idx}, param={param}). "
                f"Parameter name ('{name}') is not a valid identifier."
            )
        #
        try:
            # Try to evaluate the value, if failed, use as is its string value. Maybe confusing, but simplifies
            # providing string values, e.g., model=xgboost instead of model='xgboost'.
            hp_dict[name.strip()] = eval(value)
        except NameError:
            hp_dict[name.strip()] = value
    return hp_dict


def from_mlflow(url: str) -> t.Dict:
    """Load (best) configuration of hyperparameters from a MLflow run.

    If this MLflow run is a `train` run, return its parameters.
    If this MLflow run is an `optimize` run:
       - If `best_trial.yaml` file exists, load name of the best trial from this file, and then load `params.json`
         of the corresponding ray tune trial.
       - Else, find the best ray tune run that minimized either `valid_mse` (regression) or `valid_loss_mean`
         (classification) metrics. Then, load corresponding `params.json` file.

    Args:
        url: MLflow URI (mlflow:///MLFLOW_RUN_ID).
    Returns:
        Dictionary of parameters.
    """
    import copy

    import mlflow
    from mlflow.utils.file_utils import local_file_uri_to_path
    from ray import tune
    from ray.tune.experiment.trial import Trial

    from xtime.run import RunType

    if url.startswith("mlflow:///"):
        url = url[10:]

    run = mlflow.get_run(run_id=url)
    if "run_type" not in run.data.tags:
        raise RuntimeError("Can't use MLflow run as configuration source: `run_type` tag not present.")
    run_type: str = RunType(run.data.tags["run_type"])

    if run_type == RunType.TRAIN:
        # Just use parameters of this run
        hp_dict = copy.deepcopy(run.data.params)
    elif run_type == RunType.HPO:
        # Find the best trial and return its parameters
        artifact_path: Path = Path(local_file_uri_to_path(run.info.artifact_uri))
        if (artifact_path / "best_trial.yaml").is_file():
            best_trial_info: t.Dict = IO.load_dict(artifact_path / "best_trial.yaml")
            best_trial_path = artifact_path / best_trial_info["relative_path"]
        else:
            experiment = tune.ExperimentAnalysis((artifact_path / "ray_tune").as_posix())
            dataset_info: t.Dict = IO.load_dict(artifact_path / "dataset_info.yaml")
            best_trial: t.Optional[Trial] = experiment.get_best_trial(
                "valid_mse" if dataset_info["Dataset"]["task"] == "REGRESSION" else "valid_loss_mean", mode="min"
            )
            if best_trial is None:
                raise RuntimeError(
                    f"Can't identify best trial to retrieve hyperparameters (run_type={run_type}, "
                    f"run_id={run.info.run_id}, status={run.info.status})."
                )
            best_trial_path = Path(best_trial.logdir)
        hp_dict = from_file(best_trial_path / "params.json")
    else:
        raise RuntimeError(f"Can't use MLflow run as configuration source: unsupported run_type={run_type}.")

    return hp_dict


def from_file(url: PathLike) -> t.Dict:
    """Load dictionary of hyperparameters from a file.

    Args:
        url: Path to a file that contains python dictionary. One of supported formats are JSON and YAML.
    Returns:
        Python dictionary.
    """
    hp_dict: t.Dict = IO.load_dict(url)
    assert isinstance(hp_dict, dict), f"IO.load_dict did not return dictionary (type={type(hp_dict)})."
    return hp_dict


def _str_content(str_val: t.Optional[str], scheme: str) -> str:
    assert str_val is None or isinstance(str_val, str), f"Invalid `str_val` type ({type(str_val)})."
    str_val = str_val.strip() if isinstance(str_val, str) else ""
    if str_val.startswith(scheme):
        str_val = str_val[len(scheme) :]
    return str_val
