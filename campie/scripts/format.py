#!/usr/bin/env python3
"""This script formats/lints the entire campie source tree."""

import subprocess


def run(*args):
    print(f"> {' '.join(args)}")
    subprocess.run(args)
    print("")


def main():
    run("ruff", "--fix", ".")
    run("black", ".")
    run("isort", ".")


if __name__ == "__main__":
    main()
