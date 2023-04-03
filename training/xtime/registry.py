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

import importlib
import inspect
import typing as t
from pathlib import Path

__all__ = ["ClassRegistry"]


class ClassRegistry(object):
    """General purpose registry for classes that can automatically discover classes in a given path.

    Args:
        base_cls: Fully qualified name of the base class for all classes that are supposed to be manged by this registry
        path: Path to the directory where the classes are located
        module: Name of the python module where the classes are loaded to.

    Example:
        >>> _registry = ClassRegistry(
        ...     base_cls=f"xtime.datasets.DatasetBuilder",   # Register only classes that inherit from DatasetBuilder.
        ...     path=Path(__file__).parent,                  # Search for classes in this directory.
        ...     module="xtime.datasets"                      # Load the classes into this namespace.
        ...)

    Each class is registered under its unique name. This name is not a class name and serves as a unique identifier.
    """

    def __init__(self, base_cls: str, path: Path, module: str) -> None:
        self._base_cls = base_cls
        self._path = path.resolve()
        self._module = module
        self._registry: t.Optional[t.Dict[str, t.Type]] = None

    def contains(self, name: str) -> bool:
        """Check if a class with the given name is registered in the registry."""
        self._maybe_init()
        return name in self._registry

    def get(self, name: str):
        """Return class registered under the given name."""
        self._maybe_init()
        return self._registry.get(name)

    def keys(self) -> t.List[str]:
        self._maybe_init()
        return list(self._registry.keys())

    def _maybe_init(self) -> None:
        if self._registry is None:
            self._init()

    def _init(self) -> None:
        base_cls_module, base_cls_name = self._base_cls.rsplit(".", maxsplit=1)
        base_cls_type = getattr(importlib.import_module(base_cls_module), base_cls_name)
        self._registry = {}
        for file_path in self._path.glob("*.py"):
            module = importlib.import_module(f"{self._module}.{file_path.name[:-3]}")
            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, base_cls_type)
                    and obj is not base_cls_type
                    and getattr(obj, "NAME", None) is not None
                ):
                    for _name in [obj.NAME] if isinstance(obj.NAME, str) else obj.NAME:
                        self._register(_name, obj, module)

    def _register(self, name: str, registrable: t.Any, module) -> None:
        if name in self._registry:
            raise ValueError(
                f"Object with name ({name}) from module {module} already exists in registry ({self._registry[name]}). "
                f"Registry keys: {self.keys()}."
            )
        self._registry[name] = registrable
