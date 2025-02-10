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
import importlib.util
import logging
import os
import typing as t
from dataclasses import dataclass, field
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from pandas.core.dtypes.common import is_categorical_dtype

from xtime.errors import DatasetError
from xtime.io import IO
from xtime.ml import ClassificationTask, Feature, FeatureType, RegressionTask, Task
from xtime.registry import ClassRegistry

__all__ = [
    "DatasetSplit",  # Train, test, valid or any other splits without metadata attached.
    "DatasetMetadata",  # Metadata about concrete dataset.
    "Dataset",  # Combines metadata with multiple splits into one structure.
    "DatasetBuilder",  # Class that builds one or multiple versions of a particular dataset.
    "DatasetFactory",  # Abstract dataset factory.
    "SerializedDatasetFactory",  # Factory creates datasets that were previously serialized on disk.
    "RegisteredDatasetFactory",  # Factory creates datasets that are implemented in child classes of `DatasetBuilder`.
    "DatasetPrerequisites",  # Standard checks for dataset prerequisites.
    "DatasetTestCase",
]

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """A dataset for one Machine Learning split (train/eval/test etc.)."""

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
    """Dataset metadata that includes dataset name and version, task, features information and user properties."""

    name: str
    version: str
    task: Task
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
    """Dataset includes its metadata and splits."""

    metadata: DatasetMetadata
    splits: t.Dict[str, DatasetSplit] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        # This string will be empty if there are no errors.
        error_msg: str = ""
        # These features are expected to be present in dataset splits.
        expected_features: t.Set[str] = set([f.name for f in self.metadata.features])
        for split_name, split_data in self.splits.items():
            # These are the actual features in this dataset split.
            actual_features: t.Set[str] = set(split_data.x.columns)
            # Features that are not present in current split.
            missing_features = expected_features - actual_features
            if missing_features:
                error_msg += f" Missing features ({missing_features}) in '{split_name}'."
            # Features in split that are not described in dataset metadata.
            unknown_features = actual_features - expected_features
            if unknown_features:
                error_msg += f" Unknown features ({unknown_features}) in '{split_name}'."

        error_msg = error_msg.strip()
        if error_msg:
            raise ValueError(f"Invalid dataset specification. {error_msg}")

        # Check labels have int32 data type
        if self.metadata.task.type.classification():
            for split_name, split_data in self.splits.items():
                if split_data.y.dtype != np.int32:
                    # TODO sergey: raise an exception once unit tests are added for all datasets.
                    # raise ValueError(f"Labels in '{split_name}' split should have int32 data type.")
                    logger.warning("Labels in '%s' split should have int32 data type.", split_name)

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

    def summary(self, verbose: bool = False) -> t.Dict:
        """Return summary of the datasets.

        Args:
            verbose: If true, return detailed information about the data set splits, else, return brief summary.
        Returns:
            A dictionary with two fields. The `metadata` field contains json-serializable dataset metadata. The
            `splits` field contains information about dataset splits. This will be either brief or detailed information.
        """
        summary = {"metadata": self.metadata.to_json(), "splits": {}}
        if not verbose:
            # Provide brief description about splits
            for name, split in self.splits.items():
                y_shape: t.Optional[t.List[int]] = list(split.y.shape) if split.y is not None else None
                summary["splits"][name] = {"x": list(split.x.shape), "y": y_shape}
        else:
            # Provide detailed description about splits
            def _cats_to_str(_feat_values: pd.Series) -> str:
                _value_counts: pd.Series = _feat_values.value_counts().sort_index()
                if len(_value_counts) <= 4:
                    return str(_value_counts.to_dict())
                _first: str = str(_value_counts[:2].to_dict())
                _last: str = str(_value_counts[-2:].to_dict())
                return _first[:-1] + ", ..., " + _last[1:]

            for name, split in self.splits.items():
                summary["splits"][name] = []
                if isinstance(split.y, pd.Series):
                    col_info = {"name": name, "type": "target", "dtype": str(split.y.dtype)}
                    if self.metadata.task.type.classification():
                        col_info["details"] = f"classes={_cats_to_str(split.y)}"
                    else:
                        col_info["details"] = f"min={split.y.min()};max={split.y.max()}"
                    summary["splits"][name].append(col_info)
                for feat_name, feat_dtype in zip(split.x.columns, split.x.dtypes):
                    feat_values: pd.Series = split.x[feat_name]
                    col_info = {"name": feat_name, "type": "feature", "dtype": str(feat_dtype)}
                    if is_categorical_dtype(feat_dtype):
                        col_info["details"] = f"categories={_cats_to_str(feat_values)}"
                    else:
                        col_info["details"] = (
                            f"min={feat_values.min()};median={feat_values.median()};" f"max={feat_values.max()}"
                        )
                    summary["splits"][name].append(col_info)

        return summary

    def save(self, directory: Path = Path.cwd(), overwrite: bool = False) -> None:
        """Save dataset to disk.

        Args:
            directory: Directory where to save the dataset.
            overwrite: If true, overwrite existing files. If true, and files (metadata.yaml and/or pickle files) exist,
                they **all** will be removed.
        """
        import pickle

        from xtime.io import IO

        directory = directory.absolute()
        directory.mkdir(parents=True, exist_ok=True)

        # Check if files already exist - in current implementation this may impact subsequent dataset loading calls.
        metadata_file_exists: bool = (directory / "metadata.yaml").is_file()
        pickle_files = list(directory.glob("*.pickle"))
        if metadata_file_exists or pickle_files:
            msg: str = f"Dataset may already be present in '{directory}'."
            if metadata_file_exists:
                msg += " Metadata file (metadata.yaml) has been found."
            if pickle_files:
                msg += f" Pickle files ({[f.name for f in pickle_files]}) have been found."

            if not overwrite:
                msg += (
                    " Remove these files first, or call the `save` method with `overwrite=True` "
                    "to remove these files automatically. "
                )
                raise FileExistsError(msg)

            msg += (
                " Since overwrite is True, I will remove existing existing file (s) since they **will** impact "
                "the subsequent dataset loading calls."
            )
            logger.warning(msg)

            if metadata_file_exists:
                (directory / "metadata.yaml").unlink()
            for pickle_file in pickle_files:
                pickle_file.unlink()

        def _save_split(_ds: DatasetSplit, _split_name: str) -> None:
            _file_path = directory / f"{_split_name}.pickle"
            if not _file_path.exists():
                logger.debug("Saving %s's %s split.", self.metadata.name, _split_name)
                with open(_file_path, "wb") as _file:
                    pickle.dump({"x": _ds.x, "y": _ds.y}, _file)
            else:
                logger.debug("The %s's %s split file exists, skipping.", self.metadata.name, _split_name)

        for name, split in self.splits.items():
            _save_split(split, name)
        IO.save_yaml(self.metadata.to_json(), directory / "metadata.yaml")

    @classmethod
    def load(cls, directory: Path) -> "Dataset":
        """Load dataset from disk.

        Args:
            directory: Directory where dataset is stored.
        Returns:
            Dataset instance.
        """
        import pickle

        if not (directory / "metadata.yaml").is_file():
            raise FileNotFoundError(f"Can't locate dataset in '{directory}' directory.")

        ds = Dataset(metadata=DatasetMetadata.from_json(IO.load_yaml(directory / "metadata.yaml")))
        for file_name in directory.glob("*.pickle"):
            split_name = file_name.stem
            with open(file_name, "rb") as file:
                data = pickle.load(file)
                ds.splits[split_name] = DatasetSplit(x=data["x"], y=data["y"])
        return ds

    @staticmethod
    def create(dataset: str, **kwargs) -> "Dataset":
        """Create a dataset.

        Args:
            dataset: Dataset specification supported by the projects. Examples include serialized datasets (dataset is
                a file path to a directory), and standard (registered datasets) in which case the `dataset` is the name
                and version (name:version)
            kwargs: If this is a registered dataset, kwargs are passed to `DatasetBuilder.build` method.
        Returns:
            Dataset instance.
        """
        factories: t.List[DatasetFactory] = DatasetFactory.resolve_factories(dataset)
        registered_datasets = sorted(RegisteredDatasetFactory.registry.keys())
        if not factories:
            raise ValueError(
                f"Dataset (dataset={dataset}) not found in the registry of datasets and not resolved as a serialized "
                f"dataset. Available datasets in registry: {registered_datasets}."
            )
        if len(factories) > 1:
            sources: str = ". ".join(factory.describe() for factory in factories)
            raise ValueError(f"The dataset (name='{dataset}') can be loaded from multiple locations. {sources}.")
        return factories[0].create(**kwargs)

    @staticmethod
    def parse_name(name: str) -> t.Tuple[str, str]:
        """Parse name and return (name, version) tuple."""
        name = name.strip()
        if not name:
            return "", ""
        if ":" not in name:
            return name, ""

        _split: t.List[str] = name.split(":", maxsplit=1)
        assert len(_split) == 2, f"Expecting list size to be 2 (size={len(_split)})."
        return _split[0], _split[1]


class DatasetBuilder(object):
    """Base class for standard datasets."""

    NAME: t.Optional[str] = None

    @staticmethod
    def _patch_minio() -> None:
        if os.environ.get("XTIME_DISABLE_PATCH_MINIO", "0") == "1":
            logger.debug("[patch_minio] patch not performed: XTIME_DISABLE_PATCH_MINIO == 1.")
            return

        proxy_url: t.Optional[str] = None
        for proxy_url_param in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
            if os.environ.get(proxy_url_param, None):
                proxy_url = os.environ[proxy_url_param]
                break
        if not proxy_url:
            logger.debug("[patch_minio] patch not performed: no [https_proxy, HTTPS_PROXY, http_proxy, HTTP_PROXY]")
            return

        import minio
        import urllib3

        if getattr(minio.Minio.__init__, "__timex_patched", None) is True:
            logger.debug("[patch_minio] patch not performed: already patched")
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
        self._check_pre_requisites()
        return self.builders[version](**kwargs)

    def _check_pre_requisites(self) -> None:
        """Check if source (raw) dataset resources exist or check that all necessary libraries are available.

        Raises:
            xtime.errors.DatasetError with error_code = xtime.errors.ErrorCode.DATASET_MISSING_PREREQUISITES_ERROR
        """
        ...

    @abc.abstractmethod
    def _build_default_dataset(self, **kwargs) -> Dataset: ...

    def _build_numerical_dataset(self, **kwargs) -> Dataset:
        kwargs = kwargs.copy()

        precision: str = kwargs.pop("precision", "double")
        if precision not in {"double", "single"}:
            raise ValueError(f"Unsupported precision type ('{precision}').")

        if kwargs:
            raise ValueError(
                f"{self.__class__.__name__}: `numerical` dataset did not consume the following parameters: {kwargs}."
            )

        dataset = self._build_default_dataset()

        for feature in dataset.metadata.features:
            feature.type = FeatureType.CONTINUOUS

        dtype = np.float64 if precision == "double" else np.float32
        dataset.metadata.version = "numerical" if precision == "double" else "numerical32"

        for split in dataset.splits.values():
            split.x = split.x.astype(dtype)

        return dataset


class DatasetFactory(abc.ABC):
    """Base class factories that are used to create dataset instances.

    Factories are responsible for creating datasets from various sources. The example of sources are serialized datasets
    on disks or standard datasets supported by the project. Each instance of a factory is responsible for creating an
    instance of a particular dataset.
    """

    def __init__(self) -> None: ...

    def describe(self) -> str:
        return ""

    @abc.abstractmethod
    def create(self, **kwargs) -> Dataset:
        raise NotImplementedError

    @staticmethod
    def resolve_factories(dataset: str) -> t.List["DatasetFactory"]:
        """Identify all factories that can create this dataset."""
        return list(
            filter(
                lambda factory: factory is not None,
                [SerializedDatasetFactory.resolve(dataset), RegisteredDatasetFactory.resolve(dataset)],
            )
        )


class SerializedDatasetFactory(DatasetFactory):
    """Factory for creating datasets previously serialized on disk.

    Args:
        dir_path: Directory path to a serialized dataset.
    """

    def __init__(self, dir_path: Path) -> None:
        super().__init__()
        self.dir_path = dir_path

    def describe(self) -> str:
        return f"Serialized dataset (path={self.dir_path.as_posix()})."

    def create(self, **kwargs) -> Dataset:
        return Dataset.load(self.dir_path)

    @classmethod
    def resolve(cls, dataset: str) -> t.Optional["SerializedDatasetFactory"]:
        """Try resolving dataset name as a previously serialized dataset.

        Args:
            dataset: A possible path to a dataset
        Returns:
            SerializedDatasetLoader instance if the name maybe pointing to a serialized dataset, None otherwise.
                It is not guaranteed that dataset path indeed points to a correct dataset.
        """
        file_path = Path(dataset)
        if file_path.is_file() and file_path.name == "metadata.yaml":
            return SerializedDatasetFactory(file_path.parent.absolute())
        if file_path.is_dir() and (file_path / "metadata.yaml").is_file():
            return SerializedDatasetFactory(file_path.absolute())
        return None


class RegisteredDatasetFactory(DatasetFactory):
    """Factory for creating standard datasets.

    Args:
        name: Dataset name.
        version: Dataset version.
    """

    registry = ClassRegistry(
        base_cls="xtime.datasets.DatasetBuilder", path=Path(__file__).parent, module="xtime.datasets"
    )

    def __init__(self, name: str, version: str) -> None:
        super().__init__()
        self.name = name
        self.version = version

    def describe(self) -> str:
        return f"Registered dataset (name={self.name}, version={self.version})."

    def create(self, **kwargs) -> Dataset:
        return self.registry.get(self.name)().build(self.version, **kwargs)

    @classmethod
    def resolve(cls, dataset: str) -> t.Optional["RegisteredDatasetFactory"]:
        name, version = Dataset.parse_name(dataset)
        if not name:
            return None
        version = version or "default"
        if cls.registry.contains(name) and cls.registry.get(name)().version_supported(version):
            return RegisteredDatasetFactory(name, version)
        return None


class DatasetPrerequisites:
    """Various standard checks for dataset prerequisites."""

    @staticmethod
    def _get_install_lib_help(lib_name: str, groups: t.Union[str, t.List[str]]) -> str:
        help_msg = (
            "The `%s` library is an extra dependency of this project belonging to the `%s` group(s). "
            "With `poetry`, install it by running either (as an example) `poetry install --extras %s` or "
            "`poetry install --all-extras` commands."
        )
        if isinstance(groups, str):
            groups = [groups]
        return help_msg.format(lib_name, str(groups), groups[0])

    @staticmethod
    def check_openml(dataset_name: str, openml_url: str) -> None:
        """Check if the `openml` library is installed.

        Args:
            dataset_name: Name of a dataset.
            openml_url: An OpenML dataset URL
        """
        if importlib.util.find_spec("openml") is None:
            install_help_msg = DatasetPrerequisites._get_install_lib_help("openml", ["openml", "datasets", "all"])
            error_msg = "The `{}` dataset is an OpenML dataset ({}), and the `openml` library needs to be installed. {}"
            raise DatasetError.missing_prerequisites(error_msg.format(dataset_name, openml_url, install_help_msg))

    @staticmethod
    def check_tsfresh(dataset_name: str) -> None:
        if importlib.util.find_spec("tsfresh") is None:
            install_help_msg = DatasetPrerequisites._get_install_lib_help("tsfresh", ["timeseries", "datasets", "all"])
            raise DatasetError.missing_prerequisites(
                "The `{}` dataset requires `tsfresh` library to compute machine learning features. {} Once it is "
                "installed, there may be incompatible CUDA runtime found (see if the cause for the import error is "
                "`numba.cuda.cudadrv.error.NvvmSupportError` exception) - this may occur because `tsfresh` depends on "
                "`stumpy` that depends on `numba` that detects CUDA runtime and tries to use it if available. Try "
                "disabling CUDA for numba by exporting NUMBA_DISABLE_CUDA environment variable "
                "(https://numba.pydata.org/numba-doc/dev/reference/envvars.html#envvar-NUMBA_DISABLE_CUDA): "
                "`export NUMBA_DISABLE_CUDA=1`.".format(dataset_name, install_help_msg)
            )


class DatasetTestCase(TestCase):
    NAME: t.Optional[str] = None
    CLASS: t.Optional[t.Type[DatasetBuilder]] = None
    DATASETS: t.List[t.Dict] = []

    @staticmethod
    def standard(version: str, common_params: t.Dict) -> t.Dict:
        if version not in ("default", "numerical", "numerical32"):
            raise ValueError(f"Non-standard version: {version}")
        params = {
            "version": version,
            "test_cases": [DatasetTestCase._test_consistency, DatasetTestCase._test_splits],
            **common_params,
        }
        if version == "default":
            params["test_cases"].append(DatasetTestCase._test_default_dataset)
        elif version == "numerical":
            params["test_cases"].append(DatasetTestCase._test_numerical_dataset)
        else:
            params["test_cases"].append(DatasetTestCase._test_numerical32_dataset)
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
        name, version = Dataset.parse_name(fully_qualified_name)
        self.assertIsNotNone(name, f"Dataset name is none (fully_qualified_name={fully_qualified_name}).")
        dataset_builder_cls: t.Type = RegisteredDatasetFactory.registry.get(name)
        self.assertIs(
            dataset_builder_cls,
            self.CLASS,
            f"fully_qualified_name={fully_qualified_name}, name={name}, version={version}, "
            f"dataset_builder_cls={dataset_builder_cls}, _registry.keys()={RegisteredDatasetFactory.registry.keys()}",
        )
        return dataset_builder_cls().build(version), name, version

    @staticmethod
    def _test_consistency(self: "DatasetTestCase", ds: Dataset, params: t.Dict) -> None:
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(ds.metadata.name, self.CLASS.NAME)
        self.assertEqual(ds.metadata.version, params["version"])
        if ds.metadata.task.type.classification():
            self.assertIsInstance(ds.metadata.task, ClassificationTask)
            cl_task: ClassificationTask = t.cast(ClassificationTask, ds.metadata.task)
            self.assertEqual(cl_task.num_classes, params["num_classes"])
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

    @staticmethod
    def _test_numerical32_dataset(self: "DatasetTestCase", ds: Dataset, params: t.Dict) -> None:
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
                    split.x[col].dtype == np.float32,
                    f"Not a float32 dtype: col={col}, dtype={split.x[col].dtype}",
                )
