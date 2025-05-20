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

import tempfile
import typing as t
from pathlib import Path
from unittest import TestCase

import ray.tune as tune

import xtime.hparams as hp
from xtime.io import IO


class TestUtils(TestCase):
    def _save_example_hparams_config_to_file(self, save_dir: str, file_name: str, extension: str = "yaml") -> Path:
        hparams_file = Path(save_dir) / f"{file_name}.{extension}"
        IO.save_yaml(
            {
                "max_leaves": 256,
                "learning_rate": 0.01,
                "verbose": True,
                "msg": "hello",
                "max_depth": "ValueSpec(int, 6, tune.randint(1, 11))",
            },
            hparams_file,
        )
        return hparams_file

    def _check_randint_value_spec(self, params: t.Dict) -> None:
        self.assertIsInstance(params, dict)
        self.assertEqual(1, len(params))
        self.assertIn("max_depth", params)

        value: hp.ValueSpec = params["max_depth"]
        self.assertIsInstance(value, hp.ValueSpec)
        self.assertIs(int, value.dtype)
        self.assertEqual(6, value.default)

        for _ in range(10):
            rv: int = value.space.sample()
            self.assertIsInstance(rv, int)
            self.assertTrue(1 <= rv < 11)

    def test_from_none(self) -> None:
        self.assertDictEqual({}, hp.get_hparams())
        for hp_specs in (None, [], [None], [None, None], (None,), (None, None)):
            self.assertDictEqual({}, hp.get_hparams(hp_specs))

    def test_from_dict(self) -> None:
        self.assertDictEqual({}, hp.get_hparams({}))
        self.assertDictEqual({"a": 1}, hp.get_hparams({"a": 1}))
        self.assertDictEqual({"a": 1, "b": 2}, hp.get_hparams({"a": 1, "b": 2}))
        params: t.Dict = hp.get_hparams({"max_depth": hp.ValueSpec(int, 6, tune.randint(1, 11))})
        self._check_randint_value_spec(params)

    def test_from_string(self) -> None:
        self.assertDictEqual({}, hp.get_hparams(""))

        self.assertDictEqual(
            {"max_leaves": 256, "max_depth": 6, "learning_rate": 0.01, "verbose": True, "msg": "hello"},
            hp.get_hparams("max_leaves=256;max_depth=6;learning_rate=0.01;verbose=True;msg='hello'"),
        )

        params: t.Dict = hp.get_hparams("max_depth=ValueSpec(int, 6, tune.randint(1, 11))")
        self._check_randint_value_spec(params)

    def test_from_params_protocol(self) -> None:
        self.assertDictEqual(
            {"n_streams": 4, "n_estimators": 1, "n_trees": 1},
            hp.from_string("params:n_streams=4;n_estimators=1;n_trees=1"),
        )
        self.assertDictEqual(
            {"n_streams": 4, "n_estimators": 1, "n_trees": 1},
            hp.from_string("params:;n_streams=4;n_estimators=1;n_trees=1;"),
        )
        with self.assertRaises(ValueError) as assert_raises_context:
            _ = hp.from_string("n_streams=4;params:n_estimators=1;n_trees=1;")
        msg = str(assert_raises_context.exception)
        self.assertTrue(msg.endswith("Parameter name ('params:n_estimators') is not a valid identifier."))

    def test_from_file(self) -> None:
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            hparams_file: Path = self._save_example_hparams_config_to_file(tmp_dir, "hparams", "yaml")

            hparams: t.Dict = hp.get_hparams(f"file:{hparams_file.as_posix()}")
            self.assertIsInstance(hparams, dict)
            self.assertIn("max_depth", hparams)

            max_depth: hp.ValueSpec = hparams.pop("max_depth")
            self._check_randint_value_spec({"max_depth": max_depth})

            self.assertDictEqual({"max_leaves": 256, "learning_rate": 0.01, "verbose": True, "msg": "hello"}, hparams)

    def test_from_multiple_sources(self) -> None:
        tmp_dir: str
        with tempfile.TemporaryDirectory() as tmp_dir:
            hparams_file: Path = self._save_example_hparams_config_to_file(tmp_dir, "hparams", "yaml")
            hparams: t.Dict = hp.get_hparams(
                [None, "", f"file:{hparams_file.as_posix()}", {}, "msg='hello_2'", {"a": 100}]
            )
            self.assertIsInstance(hparams, dict)
            self.assertIn("max_depth", hparams)

            max_depth: hp.ValueSpec = hparams.pop("max_depth")
            self._check_randint_value_spec({"max_depth": max_depth})

            self.assertDictEqual(
                {"max_leaves": 256, "learning_rate": 0.01, "verbose": True, "msg": "hello_2", "a": 100}, hparams
            )
