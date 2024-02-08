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

import logging
import sys
import typing as t

import mlflow
from omegaconf import DictConfig, OmegaConf

import xtime.contrib.tune_ext as ray_tune_extensions
from xtime.contrib.mlflow_ext import MLflow
from xtime.contrib.utils import Text, log_deprecate_msg_for_run_inputs
from xtime.datasets import build_dataset
from xtime.estimators import Estimator, get_estimator
from xtime.hparams import HParamsSource, default_hparams, get_hparams
from xtime.io import IO, encode
from xtime.run import Context, Metadata, RunType

logger = logging.getLogger(__name__)

__all__ = ["run", "create_example_config"]


def run(config: DictConfig) -> None:
    config = OmegaConf.merge(_DEFAULT_CONFIGURATION, config)
    assert config.stage == "train", f"Invalid stage {config.stage}."

    ray_tune_extensions.add_representers()
    experiment_id: t.Optional[str] = MLflow.create_experiment(name=config.mlflow.experiment_name)

    description: Text = Text.from_chunks(config.mlflow.description, " ".join(sys.argv))
    with mlflow.start_run(description=str(description), experiment_id=experiment_id) as active_run:
        MLflow.init_run(active_run, set_tags_from_env=True, user_tags=config.mlflow.tags)

        log_deprecate_msg_for_run_inputs(logger)
        IO.save_yaml(
            data=encode({"dataset": config.dataset, "model": config.model, "hparams": config.hparams}),
            file_path=MLflow.get_artifact_path(active_run) / "run_inputs.yaml",
            raise_on_error=False,
        )
        OmegaConf.save(config, MLflow.get_artifact_path(active_run) / "experiment.yaml", resolve=False)

        context = Context(
            metadata=Metadata(dataset=config.dataset, model=config.model, run_type=RunType.TRAIN),
            dataset=build_dataset(config.dataset),
        )
        hparams: HParamsSource = config.hparams
        if config.hparams is None:
            hparams = default_hparams(model=config.model, task=context.dataset.metadata.task, run_type=RunType.TRAIN)

        hp_dict: t.Dict = get_hparams(hparams)
        logger.info("Hyperparameters resolved to: '%s'", hp_dict)

        estimator: t.Type[Estimator] = get_estimator(config.model)
        _ = estimator.fit(hp_dict, context)
        print(f"MLflowRun uri=mlflow:///{active_run.info.run_id}")


def create_example_config() -> DictConfig:
    """Create a template to be used with `experiment run` command.

    Returns:
        An example configuration file for the `train` experiment.
    """
    return OmegaConf.merge(
        _DEFAULT_CONFIGURATION,
        {
            "stage": "train",
            "dataset": "churn_modelling:default",
            "model": "xgboost",
            "hparams": "auto:default:model=xgboost;task=binary_classification;run_type=train",
        },
    )


_DEFAULT_CONFIGURATION = OmegaConf.create(
    {
        "stage": "???",
        "dataset": "???",
        "model": "???",
        "hparams": None,
        "mlflow": {"experiment_name": "${oc.env:MLFLOW_EXPERIMENT_NAME, null}", "description": "", "tags": {}},
    }
)
"""Default configuration template (`???` means mandatory value, null - None)."""
