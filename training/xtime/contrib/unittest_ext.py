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
import functools
import os
import tempfile
import typing as t
from enum import Enum
from unittest import TestCase

__all__ = ["check_enum", "CurrentWorkingDirectory", "with_temp_work_dir"]


def check_enum(test_case: TestCase, enum_cls: t.Type[Enum], enum_var: Enum, name: str, value: str) -> None:
    test_case.assertEqual(enum_var.name, name)
    test_case.assertEqual(enum_var.value, value)
    test_case.assertIs(enum_cls(enum_var), enum_var)
    test_case.assertIs(enum_cls(value), enum_var)


class CurrentWorkingDirectory:
    """Temporarily change current working directory.

    Args:
        directory: Working directory to set.
    """

    def __init__(self, directory: str):
        self._new_directory = directory
        self._old_directory: t.Optional[str] = None

    def __enter__(self):
        self._old_directory = os.getcwd()
        os.chdir(self._new_directory)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._old_directory)


def with_temp_work_dir(fn: t.Callable) -> t.Callable:
    """Decorator to run a function or method (for test cases) in a temporary working directory.
    https://stackoverflow.com/a/170174/575749

    Args:
        fn: Test function to decorate.

    Returns:
        Decorated function.
    """

    @functools.wraps(fn)
    def decorated_fn(*args, **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir, CurrentWorkingDirectory(temp_dir):
            fn(*args, **kwargs)

    return decorated_fn
