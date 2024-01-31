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
import typing as t
from pathlib import Path, WindowsPath

import mlflow
import numpy as np
import pandas as pd
import requests
import yaml
from mlflow.utils.file_utils import local_file_uri_to_path

__all__ = ["encode", "PathLike", "to_path", "IO"]

logger = logging.getLogger(__name__)


PathLike = t.Union[str, Path]


def to_path(path: t.Union[str, Path], check_is_file: bool = False) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not isinstance(path, Path):
        raise ValueError(f"Unsupported path (path={path}, type={type(path)})")

    if check_is_file:
        if not path.is_file():
            raise RuntimeError(f"Path is not a file (path={path})")

    return path


def encode(obj: t.Any) -> t.Any:
    """General purpose encoder to encode object recursively to JSON serializable format."""
    if isinstance(obj, (list, tuple)):
        return [encode(item) for item in obj]
    if isinstance(obj, t.Dict):
        return {key: encode(value) for key, value in obj.items()}
    if isinstance(obj, (Path, WindowsPath)):
        return obj.as_posix()
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            return float(obj.item())
        return obj.tolist()
    if isinstance(obj, np.float64):
        return float(obj)
    return obj


class IO(object):
    @staticmethod
    def download(url: str, data_dir: Path, file_name: str) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True)
        with open(data_dir / file_name, "wb+") as file:
            for chunk in response.iter_content(8192):
                file.write(chunk)

    @staticmethod
    def get_path(path: t.Optional[t.Union[str, Path]], default_path: t.Union[str, Path]) -> Path:
        path = path or default_path
        if isinstance(path, str):
            path = Path(path)
        return path.expanduser()

    @staticmethod
    def work_dir() -> Path:
        if mlflow.active_run() is not None:
            return Path(local_file_uri_to_path(mlflow.active_run().info.artifact_uri))
        return Path.cwd()

    @staticmethod
    def load_yaml(file_path: t.Union[str, Path]) -> t.Any:
        """Load YAML file.

        Args:
            file_path: Path to a YAML file.
        Returns:
            Content of this YAML file.
        """
        with open(file_path, "r") as stream:
            return yaml.load(stream, Loader=yaml.SafeLoader)

    @staticmethod
    def load_json(file_path: t.Union[str, Path]) -> t.Any:
        """Load JSON file.

        Args:
            file_path: Path to a JSON file.
        Returns:
            Content of this JSON file.
        """
        with open(file_path, "r") as stream:
            return json.load(stream)

    @staticmethod
    def load_dict(file_path: PathLike) -> t.Dict:
        file_path = to_path(file_path)
        _dict: t.Optional[t.Union] = None
        if file_path.suffix in (".yaml", ".yml"):
            _dict = IO.load_yaml(file_path)
        elif file_path.suffix == ".json":
            _dict = IO.load_json(file_path)
        if not isinstance(_dict, t.Dict):
            raise ValueError(f"File content is not a dict (file_path={file_path}, content_type={type(_dict)})")
        return _dict

    @staticmethod
    def save_yaml(data: t.Any, file_path: t.Union[str, Path]) -> None:
        try:
            with open(file_path, "w") as stream:
                yaml.dump(data, stream, Dumper=yaml.SafeDumper)
        except yaml.representer.RepresenterError:
            logger.warning("file path = %s, data  = %s", file_path, data)
            raise

    @staticmethod
    def save_json(data: t.Any, file_path: t.Union[str, Path]) -> None:
        with open(file_path, "w") as stream:
            json.dump(data, stream)

    @staticmethod
    def save_data_frame(df: pd.DataFrame, file_path: t.Union[str, Path]) -> None:
        if file_path.endswith(".yaml"):
            IO.save_yaml(df.to_dict("records"), file_path)
        elif file_path.endswith(".json"):
            IO.save_yaml(df.to_dict("records"), file_path)
        else:
            if not file_path.endswith((".csv", ".csv.gz")):
                raise NotImplementedError(f"Do not know how to serialize data frame to `{file_path}`.")
            df.to_csv(file_path)

    @staticmethod
    def save_to_file(data: t.Any, file_name: str) -> None:
        if isinstance(data, pd.DataFrame):
            IO.save_data_frame(data, file_name)
        elif file_name.endswith(".yaml"):
            IO.save_yaml(data, file_name)
        elif file_name.endswith(".json"):
            IO.save_json(data, file_name)
        else:
            raise NotImplementedError(f"Do not know how to serialize `{type(data)}` to `{file_name}`.")

    @staticmethod
    def print(data: t.Any) -> None:
        if isinstance(data, dict):
            print(json.dumps(data, indent=4, sort_keys=True))
        elif isinstance(data, pd.DataFrame):
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", -1)
            print(data.to_string(index=False))
        else:
            raise NotImplementedError(f"Do not know how to print `{type(data)}`.")
