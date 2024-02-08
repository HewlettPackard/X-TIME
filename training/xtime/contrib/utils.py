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
import logging
import typing as t

__all__ = ["Text", "normalize_str", "log_deprecate_msg_for_run_inputs"]

from omegaconf import DictConfig, ListConfig, OmegaConf


class Text:
    """Helper class to manipulate text chunks to create text documents."""

    def __init__(self, content: t.Optional[str] = None) -> None:
        self.content = normalize_str(content)

    def __str__(self) -> str:
        return self.content

    @classmethod
    def from_chunks(cls, *chunks: t.Optional[str], sep: str = "\n") -> "Text":
        return cls(sep.join([c for c in (normalize_str(_c) for _c in chunks) if c]))


def normalize_str(s: t.Optional[str]) -> str:
    """Normalize input string.
    Args:
        s: Optional string.
    Returns:
        Empty string if s is None, else input string with removed leading and trailing whitespaces,
            new line and tab characters.
    """
    return (s or "").strip()


def log_deprecate_msg_for_run_inputs(logger: logging.Logger) -> None:
    """This is an internal temporary function that will be removed."""
    logger.warning(
        "The `run_inputs.yaml` is deprecated and will not be generated in the future versions. "
        "Start using the `experiment.yaml` file instead."
    )
