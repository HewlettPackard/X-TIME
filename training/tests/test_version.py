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
from unittest import TestCase


class TestVersion(TestCase):
    def test_version(self) -> None:
        from xtime import __version__

        self.assertIsInstance(__version__, str, f"Invalid __version__ type ({type(__version__)}).")
        self.assertFalse(__version__ == "none", "The __version__ value must not be 'none'.")
