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

import typing as t
from unittest import TestCase

import xtime.hparams as hp


class TestUtils(TestCase):
    def test_parse_params(self) -> None:
        self.assertDictEqual({}, hp.from_string())
        self.assertDictEqual({}, hp.from_string(None))
        self.assertDictEqual({}, hp.from_string(""))

        self.assertDictEqual(
            {"max_leaves": 256, "max_depth": 6, "learning_rate": 0.01, "verbose": True, "msg": "hello"},
            hp.from_string("max_leaves=256;max_depth=6;learning_rate=0.01;verbose=True;msg='hello'"),
        )

        params: t.Dict = hp.from_string("max_depth=ValueSpec(int, 6, tune.randint(1, 11))")
        self.assertIsInstance(params, dict)
        self.assertEqual(1, len(params))
        self.assertIn("max_depth", params)

        value = params["max_depth"]
        self.assertIsInstance(value, hp.ValueSpec)
        self.assertIs(int, value.dtype)
        self.assertEqual(6, value.default)
