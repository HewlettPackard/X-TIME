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
import typing as t
from unittest import TestCase

import pytest
from click import BaseCommand
from click.testing import CliRunner, Result

from xtime.datasets import DatasetBuilder, RegisteredDatasetFactory
from xtime.errors import ErrorCode
from xtime.main import (
    cli,
    dataset_describe,
    dataset_list,
    dataset_save,
    datasets,
    experiment_describe,
    experiment_search_hp,
    experiment_train,
    experiments,
    hparams,
    hparams_query,
    model_list,
    models,
)

pytestmark = pytest.mark.cli
logger = logging.getLogger(__name__)


class TestMain(TestCase):
    """python -m unittest tests.test_cli.TestMain"""

    def test_help(self) -> None:
        """python -m unittest tests.test_cli.TestMain.test_help"""
        cli_funcs = [
            cli,
            experiments,
            experiment_train,
            experiment_search_hp,
            experiment_describe,
            datasets,
            dataset_describe,
            dataset_save,
            dataset_list,
            models,
            model_list,
            hparams,
            hparams_query,
        ]
        for cli_func in cli_funcs:
            self.assertIsInstance(cli_func, BaseCommand)
            result: Result = CliRunner().invoke(cli_func, ["--help"])
            self.assertEqual(result.exit_code, 0)

    def test_dataset_describe(self) -> None:
        """python -m unittest tests.test_cli.TestMain.test_dataset_describe"""
        self.assertIsInstance(dataset_describe, BaseCommand)
        registry = RegisteredDatasetFactory.registry
        for name in registry.keys():
            dataset_builder: DatasetBuilder = registry.get(name)()
            for version in dataset_builder.builders.keys():
                result: Result = CliRunner().invoke(dataset_describe, [f"{name}:{version}"])
                if result.exit_code == ErrorCode.DATASET_MISSING_PREREQUISITES_ERROR:
                    logger.info("Dataset prerequisites are not met (%s:%s).", name, version)
                else:
                    self.assertEqual(result.exit_code, 0, f"name={name}, version={version}, output={result.output}.")

    def test_dataset_list(self) -> None:
        """python -m unittest tests.test_cli.TestMain.test_dataset_list"""
        self.assertIsInstance(dataset_list, BaseCommand)
        result: Result = CliRunner().invoke(dataset_list, [])
        self.assertEqual(result.exit_code, 0)

    def test_hparams_query(self) -> None:
        """python -m unittest tests.test_cli.TestMain.test_hparams_query"""
        self.assertIsInstance(hparams_query, BaseCommand)
        args = [
            [
                "--params",
                "params:lr=0.01;batch=tune.uniform(1, 128);n_estimators=ValueSpec(int, 100, tune.randint(100, 4001))",
            ],
            [
                "--params",
                "params:lr=0.01;batch=tune.uniform(1, 128)",
                "--params",
                "params:n_estimators=ValueSpec(int, 100, tune.randint(100, 4001))",
            ],
            ["--params", "auto:default:model=xgboost;task='multi_class_classification'"],
        ]
        for cmd_args in args:
            result: Result = CliRunner().invoke(hparams_query, cmd_args)
            self.assertEqual(result.exit_code, 0)

    def test_model_list(self) -> None:
        """python -m unittest tests.test_cli.TestMain.test_model_list"""
        self.assertIsInstance(model_list, BaseCommand)
        result: Result = CliRunner().invoke(model_list, [])
        self.assertEqual(result.exit_code, 0)

        output_lines: t.List[str] = result.output.splitlines()
        self.assertEqual(output_lines[0].strip(), "Available models:")

        available_models: t.List[str] = sorted((line[2:] for line in output_lines[1:] if line.startswith("- ")))
        self.assertListEqual(available_models, ["catboost", "dummy", "lightgbm", "rf", "xgboost"])
