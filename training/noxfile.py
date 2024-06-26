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

import os
import typing as t
from pathlib import Path

import nox
import tomlkit

XTIME_NOX_PYTHON_VERSIONS = ["3.9", "3.10", "3.11"]
"""The list of python versions to run nox sessions with. Can be overridden by setting the environment variable."""
if "XTIME_NOX_PYTHON_VERSIONS" in os.environ:
    XTIME_NOX_PYTHON_VERSIONS = os.environ["XTIME_NOX_PYTHON_VERSIONS"].split(",")


# Prevent Python from writing bytecode
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


@nox.session(python=XTIME_NOX_PYTHON_VERSIONS, name="unit")
@nox.parametrize("deps", ["pinned", "latest"])
def unit_tests(session: nox.Session, deps: str) -> None:
    """Run XTIME training unit tests.

    The `posargs` are passed through to pytest (e.g., -m datasets). Current working directory - location of the
    `noxfile.py`. On a command line, these additional args should follow after `--`. For example:
    ```
    nox -s unit-3.11(deps='pinned') -- -n 8
    ```
    where:
        `-n NUM`: number of parallel unit tests to run (with `pytest-xdist`)

    Args:
        session: Current nox session.
        deps: When pinned, all dependencies are fixed to those specified in `poetry.lock`. When latest, only primary
            dependencies specified in the pyproject.toml may be pinned. All other, secondary dependencies, may or may
            not be pinned depending on metadata of primary dependencies. When pinned dependencies are used, the poetry
            must be available externally.
    """
    # Install this project and pytest (with `pip install`).
    install_args: t.List[str] = [".[all]", "pytest", "pytest-xdist"]

    # If pinned deps to be used, export constraints from `poetry.lock` file using `poetry`.
    if deps == "pinned":
        constraints_file: str = (Path(session.create_tmp()).resolve() / "constraints.txt").as_posix()
        session.run(
            "poetry",
            "export",
            "--format=constraints.txt",
            f"--output={constraints_file}",
            "--without-hashes",
            "--with=dev",
            "--without=eda",
            "--extras=all",
            external=True,
        )
        install_args.append(f"--constraint={constraints_file}")

    session.install(*install_args)
    session.run("pytest", "-v", *session.posargs)


@nox.session()
def test_pyproject_toml(session: nox.Session) -> None:
    """Run various checks on pyproject.toml."""
    pyproject = tomlkit.parse((Path(__file__).parent / "pyproject.toml").read_bytes().decode("utf-8"))
    version = pyproject["tool"]["poetry"]["version"]
    if version != "0.0.0":
        raise ValueError(f"Invalid project version ({version}). Expected value is '0.0.0'.")
