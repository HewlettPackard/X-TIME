#!/usr/bin/env python3
"""
This script is used to generate "reference.md". It essentially traverses campie's public
API via Python's object inspection and then generates a fancy markdown document out of
the docstrings included in the source code. This minimal approach works really well for
small libraries such as campie and avoids having to bring an a large and complex 3rd
party documentation generator.
"""

import inspect
import os
from types import FunctionType, ModuleType
from typing import List, Tuple

import campie.cam
import campie.util

TOP_LEVEL_MODULE = campie
"""The top level module of the library which contains the `__all__` attribute."""

MODULES: List[ModuleType] = [
    campie.cam,
    campie.util,
]
"""
All modules that expose functions to be documented.
Each module is treated as a distinct chapter.
"""

DOCS_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../docs/reference.md")
)
"""Path to the `reference.md` that this scripts generates."""


def get_functions(module) -> list[FunctionType]:
    """
    Returns the inspection object for all functions contained in campie's public API
    """

    functions = []
    for member in inspect.getmembers(module):
        name, function = member

        # filter members down to functions
        if not inspect.isfunction(function):
            continue

        # avoid functions that are not "exported"
        if name not in TOP_LEVEL_MODULE.__all__:
            continue

        functions.append(function)

    functions.sort(key=lambda f: f.__name__)

    return functions


def get_signature_block(function: FunctionType) -> str:
    """
    Creates a markdown code block with the signature of the given function.
    While there is `inspect.signature`, this function purposefully ignores type hints
    because a) they resolve to giant unreadable strings and b) types are already given
    in the docstrings.
    """

    params = inspect.signature(function).parameters.values()
    names = []

    for param in params:
        if param.default is param.empty:
            names.append(param.name)
        else:
            names.append(f"{param.name}={param.default}")

    return f"```python\ndef {function.__name__}({', '.join(names)})\n```\n"


def generate_chapter(module: ModuleType) -> Tuple[str, str]:
    """
    Generates both the contents of the chapter for a module and
    the appropriate entry in the table of contents.
    """

    if not module.__doc__:
        raise ValueError(f"module {module.__name__} has no module-level docstring.")

    chapter = f"# `{module.__name__}`\n\n"
    chapter += f"{inspect.getdoc(module)}\n\n\n"

    toc_chapter = f"- [{module.__name__}](#{module.__name__.replace('.', '')})\n"

    functions = get_functions(module)

    for function in functions:
        if not function.__doc__:
            continue

        qualifier = f"{campie.__name__}.{function.__name__}"
        toc_chapter += f"  - [{qualifier}](#{qualifier.replace('.', '')})\n"

        chapter += f"## `{qualifier}`\n\n"
        chapter += f"{get_signature_block(function)}\n"
        chapter += f"{inspect.getdoc(function)}\n\n\n"

    return chapter, toc_chapter


def main():
    docs = ""
    toc = "## Contents\n\n"

    for module in MODULES:
        chapter, toc_chapter = generate_chapter(module)
        docs += chapter
        toc += toc_chapter

    docs = f"{toc}\n\n{docs}"
    docs = f"# CAMPIE API reference\n\n{docs}"
    docs = docs.strip()

    with open(DOCS_FILE, "w") as f:
        f.write(docs)

    print(f"wrote {DOCS_FILE}")


if __name__ == "__main__":
    main()
