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

__version__ = None
"""XTIME training version."""


def _get_version() -> str:
    """Identify the XTIME training package version."""
    #
    import importlib.metadata
    import logging
    from pathlib import Path

    #
    logger = logging.getLogger(__file__)

    # If this is a development environment, try to get the version from the pyproject.toml and Git history. This is
    # the "single source of truth". In order this to work, the poetry with poetry_dynamic_versioning plugin must be
    # installed.
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.is_file():
        try:
            import os

            import tomlkit
            from poetry_dynamic_versioning import _get_config, _get_version

            pyproject = tomlkit.parse(pyproject_path.read_bytes().decode("utf-8"))
            config = _get_config(pyproject)

            initial_dir = Path.cwd()
            os.chdir(pyproject_path.parent.as_posix())
            try:
                version, _ = _get_version(config, name=pyproject["tool"]["poetry"]["name"])
                logger.debug("Found version (%s) via poetry_dynamic_versioning.", version)
                return version
            finally:
                os.chdir(initial_dir.as_posix())
        except (ImportError, RuntimeError) as err:
            # Will also catch ModuleNotFound
            logger.debug(f"Failed to get version via poetry_dynamic_versioning module: {err}")
    else:
        logger.debug(
            "Failed to get version via poetry_dynamic_versioning module. The pyproject.toml file (%s) not found",
            pyproject_path.as_posix(),
        )

    # If this package has been installed, try to get its version from package metadata.
    try:
        version = importlib.metadata.version("xtime-training")
        logger.debug("Found version (%s) via importlib (project installed?).", version)
        return version
    except importlib.metadata.PackageNotFoundError:
        logger.debug("Failed to get version via importlib (project not installed?).")

    # Try `xtime/_version.py` file. Is this needed? Probably, when this package is used as is (no install) by adjusting
    # python paths.
    version_py_file = Path(__file__).parent / "_version.py"
    if version_py_file.is_file():
        with open(version_py_file, "rt") as stream:
            version = stream.read().strip()
        if version:
            logger.debug("Found version (%s) in _version.py file (%s).", version, version_py_file.as_posix())
            return version
        logger.debug("Found empty version (%s) in _version.py file (%s).", version, version_py_file.as_posix())
    else:
        logger.debug("Failed to get version in _version.py file (file not found).")

    return "none"


if __version__ is None:
    __version__ = _get_version()
