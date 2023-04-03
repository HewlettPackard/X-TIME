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

__all__ = [
    "HParamsSource",
    "HParams",
    "HParamsSpace",
    "JsonEncoder",
    "ValueSpec",
    "HParamsSpec",
    "get_hparams",
    "from_mlflow",
    "from_list",
    "from_string",
    "from_file",
]

from xtime.run import Context, RunType

HParamsSource = t.Union[t.Dict, str, t.Tuple[str], t.List[str]]

HParams = t.Dict
"""Dictionary of hyperparameters. It should map from a parameter name to a parameter (primitive) value."""

HParamsSpace = t.Dict
"""Hyperparameter search space (maybe, framework-specific). But can also map a parameter name to just its value."""


class JsonEncoder(json.JSONEncoder):
    """Custom JSON encoder to support additional types."""

    def default(self, obj: t.Any) -> t.Any:
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
    """Helper class to define hyperparameters that provide default value, search space and random sample."""

    def __init__(self, **kwargs) -> None:
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

    def merge(self, *args: t.Iterable[t.Dict], use_default: bool = True) -> t.Dict:
        _params = self.default() if use_default else {}
        for arg in args:
            _params.update(arg or {})
        for name in _params.keys():
            if name in self.params and _params[name] is not None:
                _params[name] = self.params[name].dtype(_params[name])
        return _params


def get_hparams(source: t.Optional[HParamsSource] = None, ctx: t.Optional[Context] = None) -> t.Dict:
    hp: t.Optional[t.Dict] = None
    if source is None:
        hp = {}
    elif isinstance(source, t.Dict):
        hp = copy.deepcopy(source)
    elif isinstance(source, (t.List, t.Tuple)):
        hp: t.Dict = {}
        for one_source in source:
            hp.update(get_hparams(one_source, ctx))
    elif isinstance(source, str):
        source = source.strip()
        if source == "default":
            if ctx is None:
                hp = {}
            else:
                from tinydb import Query

                from xtime.hparams.recommender import DefaultRecommender

                print(ctx.metadata.model, ctx.dataset.metadata.task)
                q = Query()
                q = q.tags.model == ctx.metadata.model and q.tags.tasks.any([ctx.dataset.metadata.task.type.value])
                recommender = DefaultRecommender()

                if ctx.metadata.run_type == RunType.TRAIN:
                    recommendations: t.List[t.Dict] = recommender.recommend_default_values(q)
                elif ctx.metadata.run_type == RunType.HPO:
                    recommendations: t.List[t.Dict] = recommender.recommend_search_space(q)
                else:
                    raise ValueError(f"Unknown run_type={ctx.metadata.run_type}")

                hp = recommendations[0] if recommendations else {}
        elif source.startswith("mlflow:///"):
            hp = from_mlflow(source)
        elif source.startswith("params:"):
            hp = from_string(source)
        elif source.startswith("file:") or os.path.isfile(source):
            hp = from_file(source)

    if not isinstance(hp, dict):
        raise ValueError(
            f"Unsupported source of hparams (source=(value={source}, type={type(source)}). Extracted hyperparameters "
            f"expected to be a dictionary but hp=(value={hp}, type={type(hp)})."
        )
    return hp


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
    if not params:
        return {}
    if params.startswith("params:"):
        params = params[7:]
    return from_list(params.split(";"))


def from_list(params: t.Optional[t.List[str]] = None) -> t.Dict:
    if not params:
        return {}

    # These imports may be required by the `eval` call below.
    import math  # noqa # pylint: disable=unused-import

    from ray import tune  # noqa # pylint: disable=unused-import

    from xtime.hparams import ValueSpec  # noqa # pylint: disable=unused-import

    parsed = {}
    for idx, param in enumerate(params):
        try:
            name, value = param.split("=")
        except ValueError as err:
            raise ValueError(f"Invalid parameter in from_list (params={params}, idx={idx}, param={param}).") from err
        parsed[name.strip()] = eval(value)
    return parsed


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
    run_type = RunType(run.data.tags["run_type"])

    if run_type == RunType.TRAIN:
        return copy.deepcopy(run.data.params)

    if run_type == RunType.HPO:
        artifact_path: Path = Path(local_file_uri_to_path(run.info.artifact_uri))
        if (artifact_path / "best_trial.yaml").is_file():
            best_trial = IO.load_yaml(artifact_path / "best_trial.yaml")
            return IO.load_json(artifact_path / best_trial["relative_path"] / "params.json")
        else:
            experiment = tune.ExperimentAnalysis((artifact_path / "ray_tune").as_posix())
            dataset_info = IO.load_yaml(artifact_path / "dataset_info.yaml")
            perf_metric = "valid_mse" if dataset_info["Dataset"]["task"] == "REGRESSION" else "valid_loss_mean"
            best_trial: Trial = experiment.get_best_trial(perf_metric, mode="min")
            return IO.load_json((Path(best_trial.logdir) / "params.json").as_posix())

    raise RuntimeError(f"Can't use MLflow run as configuration source: unsupported run_type={run_type}.")


def from_file(url: PathLike) -> t.Dict:
    return IO.load_dict(url)
