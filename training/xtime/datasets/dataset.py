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

import abc
import copy
import os
import typing as t
from dataclasses import dataclass, field
from pathlib import Path
from unittest import TestCase

import pandas as pd
from pandas import CategoricalDtype

from xtime.io import IO
from xtime.ml import ClassificationTask, Feature, FeatureType, RegressionTask, Task
from xtime.registry import ClassRegistry

__all__ = [
    "DatasetSplit",
    "DatasetMetadata",
    "Dataset",
    "parse_dataset_name",
    "build_dataset",
    "get_known_unknown_datasets",
    "get_dataset_builder_registry",
    "DatasetBuilder",
    "DatasetTestCase",
]


@dataclass
class DatasetSplit:
    """A dataset for one Machine Learning step (train/eval/test etc.)."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    EVAL_SPLITS = ("valid", "test")

    x: pd.DataFrame
    y: t.Optional[t.Union[pd.DataFrame, pd.Series]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.x, pd.DataFrame):
            raise TypeError(
                f"Dataset independent variables (x) should be pandas data frame (actual type = {type(self.x)})."
            )
        if self.y is not None:
            if not isinstance(self.y, (pd.DataFrame, pd.Series)):
                raise TypeError(
                    "Dataset dependent variables (y) should be pandas data frame or series "
                    f"(actual type = {type(self.y)})."
                )
            if self.x.shape[0] != self.y.shape[0]:
                raise ValueError(
                    f"Number of rows in x and y do not match (x.shape = {self.x.shape}, y.shape = {self.y.shape})"
                )


@dataclass
class DatasetMetadata:
    name: str
    version: str
    task: t.Optional[t.Union[ClassificationTask, RegressionTask]] = None
    features: t.List[Feature] = field(default_factory=lambda: [])
    properties: t.Dict[str, t.Any] = field(default_factory=lambda: {})

    def num_features(self) -> int:
        return len(self.features)

    def feature_names(self) -> t.List[str]:
        return [feature.name for feature in self.features]

    def categorical_feature_names(self) -> t.List[str]:
        return [feature.name for feature in self.features if feature.type.categorical()]

    def has_categorical_features(self) -> bool:
        return any(feature.type.categorical() for feature in self.features)

    def to_json(self) -> t.Dict:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task.to_json(),
            "features": [f.to_json() for f in self.features],
            "properties": copy.deepcopy(self.properties),
        }

    @classmethod
    def from_json(cls, json_dict: t.Dict) -> "DatasetMetadata":
        return cls(
            name=json_dict["name"],
            version=json_dict["version"],
            task=Task.from_json(json_dict["task"]),
            features=[Feature.from_json(f) for f in json_dict["features"]],
            properties=copy.deepcopy(json_dict["properties"]),
        )


@dataclass
class Dataset:
    metadata: DatasetMetadata
    splits: t.Dict[str, DatasetSplit] = field(default_factory=lambda: {})

    def num_examples(self, split_names: t.Optional[t.Union[str, t.Iterable[str]]] = None) -> int:
        if not split_names:
            split_names = self.splits.keys()
        elif isinstance(split_names, str):
            split_names = [split_names]
        return sum(self.splits[name].x.shape[0] for name in split_names)

    def split(self, split_name: t.Union[str, t.Iterable[str]]) -> t.Optional[DatasetSplit]:
        _names = [split_name] if isinstance(split_name, str) else split_name
        return next((self.splits[name] for name in _names if name in self.splits), None)

    def validate(self) -> "Dataset":
        def _validate_split(_ds: DatasetSplit, _split_name: str) -> None:
            assert isinstance(_ds, DatasetSplit), f"Bug: invalid dataset split type ({type(_ds)})."
            assert isinstance(_ds.x, pd.DataFrame), f"Unexpected type of `x` for {_split_name} dataset."
            assert isinstance(_ds.y, (pd.DataFrame, pd.Series)), f"Unexpected type of `y` for {_split_name} dataset."

        for name, split in self.splits.items():
            _validate_split(split, name)

        return self

    def summary(self) -> t.Dict:
        info: t.Dict = self.metadata.to_json()
        info["splits"] = {}
        for name, split in self.splits.items():
            info["splits"][name] = {"x": list(split.x.shape), "y": list(split.y.shape)}
        return info

    def save(self, directory: t.Optional[t.Union[str, Path]] = None) -> None:
        import pickle

        from xtime.io import IO

        directory = Path(directory or Path.cwd().as_posix()) / (self.metadata.name + "_" + self.metadata.version)
        directory.mkdir(parents=True, exist_ok=True)

        def _save_split(_ds: DatasetSplit, _split_name: str) -> None:
            _file_path = directory / f"{_split_name}.pickle"
            if not _file_path.exists():
                print(f"Saving {self.metadata.name}'s {_split_name} split.")
                with open(_file_path, "wb") as _file:
                    pickle.dump({"x": _ds.x, "y": _ds.y}, _file)
            else:
                print(f"The {self.metadata.name}'s {_split_name} split file exists, skipping.")

        for name, split in self.splits.items():
            _save_split(split, name)
        IO.save_yaml(self.metadata.to_json(), directory / "metadata.yaml")

    @staticmethod
    def load(ctx: "Context", save_info_dir: t.Optional[Path] = None) -> None:
        """Load datasets and save their information to `dataset_info.yaml` file.

        Loaded dataset object (Dataset) will be available under the `dataset` key in `context`. Any existing dataset
        will be replaced / reloaded with the new one.
        Args:
            ctx: Dataclass containing context parameters.
            save_info_dir: Path to directory where dataset information file will be serialized.

        TODO: remove thus function.
        """
        ctx.dataset = build_dataset(ctx.metadata.dataset)
        if save_info_dir is not None:
            IO.save_yaml(ctx.dataset.metadata.to_json(), save_info_dir / "dataset_info.yaml")


class DatasetBuilder(object):
    NAME: t.Optional[str] = None

    @staticmethod
    def _patch_minio() -> None:
        if os.environ.get("XTIME_DISABLE_PATCH_MINIO", "0") == "1":
            print("[patch_minio] patch not performed: XTIME_DISABLE_PATCH_MINIO == 1")
            return

        proxy_url: t.Optional[str] = None
        for proxy_url_param in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
            if os.environ.get(proxy_url_param, None):
                proxy_url = os.environ[proxy_url_param]
                break
        if not proxy_url:
            print("[patch_minio] patch not performed: no [https_proxy, HTTPS_PROXY, http_proxy, HTTP_PROXY]")
            return

        import minio
        import urllib3

        if getattr(minio.Minio.__init__, "__timex_patched", None) is True:
            print("[patch_minio] patch not performed: already patched")
            return

        def _decorate(fn: t.Callable) -> t.Callable:
            def _minio_init_wrapper(*args, **kwargs):
                if "http_client" not in kwargs:
                    kwargs["http_client"] = urllib3.ProxyManager(proxy_url=proxy_url)
                fn(*args, **kwargs)

            return _minio_init_wrapper

        minio.Minio.__init__ = _decorate(minio.Minio.__init__)
        minio.Minio.__init__.__timex_patched = True

    def __init__(self, openml: bool = False) -> None:
        self.builders: t.Dict[
            str, t.Callable[..., Dataset]  # version name  # instance builder function returning 'Dataset' instance
        ] = {}
        if openml:
            DatasetBuilder._patch_minio()

    def version_supported(self, version: str) -> bool:
        return version in self.builders

    def build(self, version: t.Optional[str] = None, **kwargs) -> Dataset:
        version = version or "default"
        if version not in self.builders:
            raise ValueError(
                f"Unrecognized dataset version: name={self.NAME}, version={version}. "
                f"Available versions: {list(self.builders.keys())}."
            )
        return self.builders[version](**kwargs)

    @abc.abstractmethod
    def _build_default_dataset(self, **kwargs) -> Dataset:
        ...

    def _build_numerical_dataset(self, **kwargs) -> Dataset:
        if kwargs:
            raise ValueError(f"{self.__class__.__name__}: `numerical` dataset does not accept arguments.")

        dataset = self._build_default_dataset()

        for feature in dataset.metadata.features:
            feature.type = FeatureType.CONTINUOUS

        for split in dataset.splits.values():
            split.x = split.x.astype(float)

        dataset.metadata.version = "numerical"
        return dataset


_registry = ClassRegistry(base_cls="xtime.datasets.DatasetBuilder", path=Path(__file__).parent, module="xtime.datasets")


def get_dataset_builder_registry() -> ClassRegistry:
    return _registry


def parse_dataset_name(name: str) -> t.Tuple[str, t.Optional[str]]:
    name_and_version = (name, None) if ":" not in name else name.split(":")
    if len(name_and_version) != 2:
        raise ValueError(f"Invalid dataset name: {name}.")
    return name_and_version


def build_dataset(name: str, **kwargs) -> Dataset:
    name, version = parse_dataset_name(name)
    version = version or "default"
    return _registry.get(name)().build(version, **kwargs)


def get_known_unknown_datasets(fully_qualified_names: t.List[str]) -> t.Tuple[t.List[str], t.List[str]]:
    known, unknown = [], []
    for fully_qualified_name in fully_qualified_names:
        name, version = parse_dataset_name(fully_qualified_name)
        if not _registry.contains(name) or not _registry.get(name)().version_supported(version or "default"):
            unknown.append(fully_qualified_name)
        else:
            known.append(fully_qualified_name)
    return known, unknown


class DatasetTestCase(TestCase):
    NAME: t.Optional[str] = None
    CLASS: t.Optional[t.Type[DatasetBuilder]] = None
    DATASETS: t.List[t.Dict] = []

    @staticmethod
    def standard(version: str, common_params: t.Dict) -> t.Dict:
        if version not in ("default", "numerical"):
            raise ValueError(f"Non-standard version: {version}")
        params = {
            "version": version,
            "test_cases": [DatasetTestCase._test_consistency, DatasetTestCase._test_splits],
            **common_params,
        }
        if version == "default":
            params["test_cases"].append(DatasetTestCase._test_default_dataset)
        else:
            params["test_cases"].append(DatasetTestCase._test_numerical_dataset)
        return params

    def _test_datasets(self) -> None:
        for params in self.DATASETS:
            ds, name, version = self._load_dataset(f"{self.NAME}:{params['version']}")

            self.assertEqual(self.NAME, name)
            self.assertEqual(self.CLASS.NAME, name)
            self.assertEqual(params["version"], version)

            for test_fn in params["test_cases"]:
                test_fn(self, ds, params)

    def _load_dataset(self, fully_qualified_name: str) -> t.Tuple[t.Any, str, str]:
        name, version = parse_dataset_name(fully_qualified_name)
        dataset_builder_cls: t.Type = _registry.get(name)
        self.assertIs(
            dataset_builder_cls,
            self.CLASS,
            f"fully_qualified_name={fully_qualified_name}, name={name}, version={version}, "
            f"dataset_builder_cls={dataset_builder_cls}, _registry.keys()={_registry.keys()}",
        )
        return dataset_builder_cls().build(version), name, version

    @staticmethod
    def _test_consistency(self: "DatasetTestCase", ds: Dataset, params: t.Dict) -> None:
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(ds.metadata.name, self.CLASS.NAME)
        self.assertEqual(ds.metadata.version, params["version"])
        if ds.metadata.task.type.classification():
            self.assertIsInstance(ds.metadata.task, ClassificationTask)
            self.assertEqual(ds.metadata.task.num_classes, params["num_classes"])
        else:
            self.assertIsInstance(ds.metadata.task, RegressionTask)
        self.assertEqual(ds.metadata.task.type, params["task"])
        self.assertEqual(params["num_features"], len(ds.metadata.features))

        self.assertEqual(len(ds.splits), len(params["splits"]))
        for split in params["splits"]:
            self.assertIn(split, ds.splits)

    @staticmethod
    def _test_splits(self: "DatasetTestCase", ds: Dataset, params: t.Dict) -> None:
        for split_name in params["splits"]:
            self.assertIn(split_name, ds.splits)

            split: DatasetSplit = ds.splits[split_name]
            self.assertIsNotNone(split)

            self.assertIsNotNone(split.x)
            self.assertIsNotNone(split.y)

            self.assertIsNotNone(split.x, pd.DataFrame)
            self.assertIsNotNone(split.y, pd.Series)
            self.assertEqual(split.x.shape[0], split.y.shape[0])
            self.assertEqual(split.x.shape[1], params["num_features"])

    @staticmethod
    def _test_default_dataset(self: "DatasetTestCase", ds: Dataset, params: t.Dict) -> None:
        for _, split in ds.splits.items():
            self.assertEqual(len(split.x.columns), len(ds.metadata.features))
            for col, feature in zip(split.x.columns, ds.metadata.features):
                self.assertEqual(col, feature.name)
                if feature.type.numerical():
                    self.assertTrue(
                        pd.api.types.is_float_dtype(split.x[col].dtype)
                        or pd.api.types.is_integer_dtype(split.x[col].dtype),
                        f"Not a float dtype: col={col}, dtype={split.x[col].dtype}",
                    )
                elif feature.type.categorical():
                    dtype: CategoricalDtype = split.x[col].dtype
                    self.assertIsInstance(
                        dtype, CategoricalDtype, f"Not a categorical dtype: col={col}, dtype={dtype}."
                    )
                    self.assertTrue(
                        pd.api.types.is_categorical_dtype(dtype), f"Not a categorical dtype: col={col}, dtype={dtype}"
                    )

                    if FeatureType.ORDINAL == feature.type:
                        self.assertTrue(dtype.ordered)
                    else:
                        self.assertFalse(dtype.ordered)
                else:
                    self.assertTrue(False, f"Unrecognized feature type: col={col}, type={feature.type}")

    @staticmethod
    def _test_numerical_dataset(self: "DatasetTestCase", ds: Dataset, params: t.Dict) -> None:
        for _, split in ds.splits.items():
            self.assertEqual(len(split.x.columns), len(ds.metadata.features))
            for col, feature in zip(split.x.columns, ds.metadata.features):
                self.assertEqual(col, feature.name)
                self.assertEqual(
                    FeatureType.CONTINUOUS,
                    feature.type,
                    f"{feature} is expected to be CONTINUOUS in numerical dataset.",
                )
                self.assertTrue(
                    pd.api.types.is_float_dtype(split.x[col].dtype),
                    f"Not a float dtype: col={col}, dtype={split.x[col].dtype}",
                )
