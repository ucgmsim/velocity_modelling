import os
from pathlib import Path

import pytest

from velocity_modelling.constants import (  # lazy resolver
    get_data_root,
    get_registry_path,
)


def env_path(var: str) -> Path | None:
    val = os.environ.get(var)
    return Path(val).expanduser().resolve() if val else None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--benchmark-dir",
        action="store",
        type=Path,
        default=Path(__file__).parent / "benchmarks",
        help="Override the default BENCHMARK_DIR directory.",
    )

    # Keep default None; resolve later in fixture with get_data_root()
    parser.addoption(
        "--data-root",
        action="store",
        type=Path,
        default=None,
        help="Path to NZCVM data root (overrides env/config/defaults).",
    )

    parser.addoption(
        "--nzvm-binary-path",
        action="store",
        type=Path,
        default=env_path("NZVM_BINARY_PATH"),
        help="Path to nzvm binary (or set NZVM_BINARY_PATH env var).",
    )
    parser.addoption(
        "--nzcvm-registry",
        action="store",
        type=Path,
        default=None,
        help="Path to nzcvm_registry.yaml (overrides the default under data root).",
    )


@pytest.fixture(scope="session")
def benchmark_dir(pytestconfig: pytest.Config) -> Path:
    p = Path(pytestconfig.getoption("--benchmark-dir")).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"Provided BENCHMARK_DIR does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"Provided BENCHMARK_DIR is not a directory: {p}")
    return p


@pytest.fixture(scope="session")
def data_root(pytestconfig: pytest.Config) -> Path:
    """Resolve NZCVM data root with precedence:
    1) --data-root (this option)
    2) NZCVM_DATA_ROOT env var
    3) ~/.config/nzcvm_data/config.json (written by `nzcvm-data install`)
    4) sensible defaults
    5) interactive prompt (disabled in tests)
    """
    cli_value: Path | None = pytestconfig.getoption("--data-root")
    resolved = get_data_root(str(cli_value) if cli_value else None)
    if not resolved.exists():
        raise FileNotFoundError(
            f"NZCVM data root not found: {resolved}. "
            "Run `nzcvm-data install`, or pass --data-root / set NZCVM_DATA_ROOT."
        )
    return resolved


@pytest.fixture(scope="session")
def nzvm_binary_path(pytestconfig: pytest.Config) -> Path | None:
    p: Path | None = pytestconfig.getoption("--nzvm-binary-path")
    return p.expanduser().resolve() if p else None


@pytest.fixture(scope="session")
def registry_path(pytestconfig: pytest.Config) -> Path:
    cli_value: Path | None = pytestconfig.getoption("--nzcvm-registry")
    rp = cli_value.expanduser().resolve() if cli_value else get_registry_path()
    if not rp.exists():
        raise FileNotFoundError(f"NZCVM registry file not found: {rp}")
    return rp
