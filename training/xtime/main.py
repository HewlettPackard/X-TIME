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

import json
import logging
import sys
import typing as t
from multiprocessing import Process
from pathlib import Path

import click
import coloredlogs
from ray import tune

from xtime.datasets import Dataset, DatasetBuilder, DatasetFactory, RegisteredDatasetFactory
from xtime.errors import XTimeError

# from xtime.estimators import get_estimator_registry

logger = logging.getLogger(__name__)


dataset_arg = click.argument("dataset", type=str, metavar="DATASET")
dataset_arg_help = (
    "Name of a dataset is in the following format: <dataset_name>[:<version>]. The version part is optional and "
    "can be one of: (default, numerical). If not present, the `default` version is used. Pay attention that not all "
    "datasets have all versions (this depends on what features are present in the dataset)."
)


model_arg = click.argument("model", type=str, metavar="MODEL")
model_arg_help = "To get list of available ML models, run `models list` command."


params_option = click.option(
    "--params",
    "-p",
    required=False,
    multiple=True,
    help="Optional sources for hyperparameters (if not present, default model's parameters will be used, that are not "
    "optimal in any way). Valid sources - YAML and JSON files; MLflow train and hyper-parameter optimization "
    "runs (defined as MLflow URI: mlflow:///MLFLOW_RUN_ID). Hyperparameter recommendation engines (auto). If this "
    "MLflow run is a hyper-parameter optimization "
    "run, the best configuration is selected (which is defined by the smallest loss on a validation dataset).",
)


def print_err_and_exit(err: Exception) -> None:
    """Print brief information about exception and exit."""
    print(str(err))
    if not logger.root.isEnabledFor(logging.DEBUG):
        print("Rerun with --log-level=debug to get detailed information.")
    logger.debug(
        "Error encountered while executing `dataset describe` command.", exc_info=err, stack_info=True, stacklevel=-1
    )
    error_code: int = err.error_code if isinstance(err, XTimeError) else 1
    exit(error_code)


def _run_search_hp_pipeline(
    dataset: str,
    model: str,
    algorithm: str,
    hparams: t.Optional[t.Tuple[str]],
    num_search_trials: int,
    num_validate_trials: int = 0,
    gpu: float = 0,
) -> None:
    from xtime.stages.search_hp import search_hp

    """Run an ML pipeline that includes (1) hyperparameter search and (2) analysis how stable hyperparameters are."""
    mlflow_uri: str = search_hp(dataset, model, algorithm, hparams, num_search_trials, gpu)
    if num_validate_trials > 0:
        validate_hparams = [
            mlflow_uri,  # Take the best hyperparameters from this MLFlow run.
            {"random_state": tune.randint(0, int(2**32 - 1))},  # And vary random seed to validate these HPs are stable.
        ]
        search_hp(dataset, model, "random", validate_hparams, num_validate_trials, gpu)


@click.group(name="xtime", help="Machine Learning benchmarks for tabular data for XTIME project.")
@click.option(
    "--log-level",
    "--log_level",
    required=False,
    default="warning",
    type=click.Choice(["critical", "error", "warning", "info", "debug"]),
    help="Logging level is a lower-case string value for Python's logging library (see "
    "[Logging Levels]({log_level}) for more details). Only messages with this logging level or higher are "
    "logged.".format(log_level="https://docs.python.org/3/library/logging.html#logging-levels"),
)
def cli(log_level: t.Optional[str]):
    if log_level:
        log_level = log_level.upper()
        logging.basicConfig(level=log_level)
        coloredlogs.install(level=log_level)
        logging.info("cli setting log Level from CLI argument to '%s'.", log_level)
    logger.debug("cli command=%s", sys.argv)


@cli.group(
    "experiment",
    help="Experiments-related commands. They can be used for training ML models and searching for hyperparameters. "
    "The XTIME project uses MLflow for experiment tracking, so make sure to configure MLflow URI should a "
    "non-default MLflow server be used.",
)
def experiments() -> None: ...


@experiments.command(
    name="train", help=f"Train a MODEL ML model on a DATASET dataset. {dataset_arg_help} {model_arg_help}"
)
@dataset_arg
@model_arg
@params_option
def experiment_train(dataset: str, model: str, params: t.Tuple[str]) -> None:
    from xtime.stages.train import train

    try:
        # When no --params are provided, the `params` will be empty. Setting no None here
        # will enable the train function to retrieve default parameters in this case.
        train(dataset, model, params if len(params) > 0 else None)
    except Exception as err:
        print_err_and_exit(err)


@experiments.command(
    name="search_hp",
    help="Optimize a MODEL ML model on a DATASET dataset - run hyperparameter optimization experiment. "
    f"{dataset_arg_help} {model_arg_help} Available algorithms are `random` and `hyperopt`.",
)
@dataset_arg
@model_arg
@click.argument("algorithm", type=str, metavar="ALGORITHM")
@params_option
@click.option(
    "--num-search-trials",
    required=False,
    type=int,
    default=100,
    help="Number of hyperparameter optimization search (HPO) trials.",
)
@click.option(
    "--num-validate-trials",
    required=False,
    type=int,
    default=0,
    help="Number of trials to retrain a model using the best configuration found with HPO.",
)
@click.option(
    "--gpu", required=False, type=float, is_flag=False, default=0, flag_value=1,
    help="A GPU fraction to use for a single trial (a number between 0 and 1). When 0, not GPUs will be used. When 1, "
         "a single GPU will be exclusively used by a single trial."
)
def experiment_search_hp(
    dataset: str,
    model: str,
    algorithm: str,
    params: t.Tuple[str],
    num_search_trials: int,
    num_validate_trials: int = 0,
    gpu: float = 0,
) -> None:
    if not (0 <= gpu <= 1):
        print("The `--gpu` option value must be a floating point value from [0, 1].")
        exit(1)
    try:
        known_problems = dataset.split(sep=";")
        for _problem in known_problems:
            factories = DatasetFactory.resolve_factories(_problem)
            if len(factories) != 1:
                print(f"Cannot create dataset ({_problem}). Number of dataset factories is {len(factories)}.")
                exit(1)
        for _problem in known_problems:
            try:
                process = Process(
                    target=_run_search_hp_pipeline,
                    # When no --params are provided, the `params` will be empty. Setting no None here
                    # will enable the search_hp function to retrieve default parameters in this case.
                    args=(
                        _problem,
                        model,
                        algorithm,
                        params if len(params) > 0 else None,
                        num_search_trials,
                        num_validate_trials,
                        gpu,
                    ),
                )
                process.start()
                process.join()
            except Exception as err:
                print(f"Error executing the `optimize` task for {dataset}. Error = {err}.")
    except Exception as err:
        print_err_and_exit(err)


@experiments.command(
    name="describe",
    help="Summarize one or multiple train/optimize MLflow runs. The REPORT_TYPE argument defines the output format." "",
)
@click.argument(
    "report_type", required=True, metavar="REPORT_TYPE", type=click.Choice(["summary", "best_trial", "final_trials"])
)
@click.option(
    "--run",
    "-r",
    metavar="MLFLOW_RUN",
    required=False,
    type=str,
    default=None,
    help="MLflow run ID or MLflow URI. This argument is mandatory when report type is one of (summary, best_trial). "
    "The value if this parameter is either a run ID or a string in the following format: mlflow:///MLFLOW_RUN_ID.",
)
@click.option(
    "--file",
    "-f",
    metavar="FILE",
    required=False,
    type=str,
    default=None,
    help="Optional file name for report. The file extension defines serialization format. If not present, generated "
    "report will be printed on a console.",
)
def experiment_describe(report_type: str, run: t.Optional[str] = None, file: t.Optional[str] = None) -> None:
    from xtime.stages.describe import describe

    try:
        describe(report_type, run, file)
    except Exception as err:
        print_err_and_exit(err)


@cli.group("dataset", help="Dataset-related commands (explore available datasets with these commands).")
def datasets() -> None: ...


@datasets.command("describe", help=f"Provide a brief DATASET dataset description. {dataset_arg_help}")
@dataset_arg
def dataset_describe(dataset: str) -> None:
    try:
        ds: Dataset = Dataset.create(dataset).validate()
        json.dump(ds.summary(), sys.stdout, indent=4)
    except Exception as err:
        print_err_and_exit(err)


@datasets.command("list", help="List all available datasets.")
def dataset_list() -> None:
    from prettytable import PrettyTable

    try:
        table = PrettyTable(field_names=["Dataset", "Versions"])
        registry = RegisteredDatasetFactory.registry
        for name in registry.keys():
            dataset_builder: DatasetBuilder = registry.get(name)()
            table.add_row([name, ", ".join(dataset_builder.builders.keys())])
        print("Available datasets:")
        print(table)
        print(dataset_arg_help)
        print("Examples:")
        print("\t- `python -m xtime.main dataset describe churn_modelling:default`")
        print("\t- `python -m xtime.main dataset describe eye_movements:numerical`")
    except Exception as err:
        print_err_and_exit(err)


@datasets.command("save", help=f"Save a DATASET dataset version on disk. {dataset_arg_help}")
@dataset_arg
@click.option(
    "--directory",
    "-d",
    required=False,
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path, resolve_path=True),
    default=None,
    metavar="DIRECTORY",
    help="Output directory where DATASET dataset is to be saved. If not specified, current working directory is used.",
    callback=lambda ctx, param, val: val if val is not None else Path.cwd(),
)
def dataset_save(dataset: str, directory: Path) -> None:
    try:
        ds: Dataset = Dataset.create(dataset).validate()
        ds.save(directory)
    except Exception as err:
        print_err_and_exit(err)


@cli.group("models", help="Machine Learning models-related commands.")
def models() -> None: ...


@models.command("list", help="List all available models.")
def model_list() -> None:
    try:
        from xtime.estimators import get_estimator_registry

        print("Available models:")
        for name in get_estimator_registry().keys():
            print(f"- {name}")
    except Exception as err:
        print_err_and_exit(err)


@cli.group("hparams", help="Hyperparameters-related commands.")
def hparams() -> None: ...


@hparams.command("query", help="Query hyperparameters for a given set of input specifications.")
@params_option
def hparams_query(params: t.Tuple[str]) -> None:
    try:
        import xtime.hparams as hp

        hp_dict: t.Dict = hp.get_hparams(params)
        json.dump(hp_dict, sys.stdout, indent=4, cls=hp.JsonEncoder)
        print("")
    except Exception as err:
        print_err_and_exit(err)


if __name__ == "__main__":
    cli()
