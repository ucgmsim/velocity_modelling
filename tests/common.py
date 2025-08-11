from pathlib import Path

import pytest


@pytest.fixture
def nzvm_c_binary_path(request: pytest.FixtureRequest) -> Path:
    """
    Get the path to the nzcvm C binary from the command-line option or environment variable.
    """
    nzvm_path = request.config.getoption("--nzvm-binary-path")
    if nzvm_path is None:
        raise ValueError(
            "nzvm binary path not provided. Use --nzvm-binary-path or NZVM_BINARY_PATH environment variable."
        )
    new_nzvm_path = Path(nzvm_path).resolve()
    if not new_nzvm_path.exists():
        raise ValueError(f"Provided nzvm binary path does not exist: {new_nzvm_path}")
    if not new_nzvm_path.is_file():
        raise ValueError(f"Provided nzvm binary path is not a file: {new_nzvm_path}")
    return new_nzvm_path
