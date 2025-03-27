import os
from pathlib import Path

import pytest

from velocity_modelling.cvm.constants import DATA_ROOT

try:
    nzcvm_binary_path = Path(
        os.environ["nzcvm_BINARY_PATH"]
    )  # check if environment variable is set
except KeyError:
    nzcvm_binary_path = (
        None  # default value. Can be overridden with --nzcvm-binary-path argument
    )


def pytest_addoption(parser: pytest.Parser):
    parser.addoption(
        "--benchmark-dir",
        action="store",
        default=Path(__file__).parent / "benchmarks",
        help="Override the default BENCHMARK_DIR directory",
    )

    parser.addoption(
        "--data-root",
        action="store",
        default=DATA_ROOT,
        help="Override the default DATA_ROOT directory",
    )

    parser.addoption(
        "--nzcvm-binary-path",
        action="store",
        default=nzcvm_binary_path,
        help="Set this or nzcvm_BINARY_PATH environment variable for the nzcvm binary path",
    )
