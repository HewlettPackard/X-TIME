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

"""
References:
- Why I Like Nox / https://hynek.me/articles/why-i-like-nox


Notes
- Anaconda setup.
  - Install nox in base environment: `conda install nox`
  - Create multiple python environments`conda create -n 3.9 python=3.9; conda create -n 3.11 python=3.11`.
    It seems like there's no need to create an environment for the same version as base environment (3.10 in this
    example).
  - Note paths to these environments: `conda env list`
  - Run nox in BASE environment: `PATH="${CONDA_PREFIX}/envs/3.9/bin:${CONDA_PREFIX}/envs/3.11/bin:${PATH}" nox`
- The script itself runs in python environment that was used to launch the script. It's the `session` object that
  communicates with the actual runtime environment.
- The `posargs` is a list of strings e.g. ["--name=val"]
- The `pip` can install poetry projects (pyproject.toml) starting 19.0 version.
- Output of `run` and `install` can be sent to a file (https://github.com/wntrblm/nox/issues/218).
"""

import os
import typing as t
from pathlib import Path

import nox

XTIME_NOX_PYTHON_VERSIONS = ["3.9", "3.10", "3.11"]
"""The list of python versions to run nox sessions with. Can be overridden by setting the environment variable."""
if "XTIME_NOX_PYTHON_VERSIONS" in os.environ:
    XTIME_NOX_PYTHON_VERSIONS = os.environ["XTIME_NOX_PYTHON_VERSIONS"].split(",")


# Prevent Python from writing bytecode
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def _print_header(session: nox.Session) -> None:
    pyver: str = session.run((Path(session.bin) / "python").as_posix(), "--version", silent=True)
    nox_info = f"(location='{session.virtualenv.location}', python='{session.python}', version='{pyver}')"
    rt_info = f"(cwd='{Path.cwd()}', posargs={session.posargs})"
    print(f"nox={nox_info}, runtime={rt_info}")


@nox.session(python=XTIME_NOX_PYTHON_VERSIONS, name="unit")
def unit_tests(session: nox.Session) -> None:
    """Run XTIME unit tests.

    The `posargs` are passed through to pytest (e.g., -m datasets). Current working directory - location of the
    `noxfile.py`.
    """
    _print_header(session)

    # Not really required, but let's use pytest version specified in project's dev dependencies.
    proj_cfg: t.Dict = nox.project.load_toml("pyproject.toml")
    dev_deps: t.Dict = proj_cfg["tool"]["poetry"]["group"]["dev"]["dependencies"]
    session.install(f"pytest=={dev_deps['pytest']}")

    # Install current project (only `main` dependencies are installed).
    session.install(".[all]")

    # Run unit tests.
    session.run("pytest", "-v", *session.posargs)
