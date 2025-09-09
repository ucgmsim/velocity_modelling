import os
from pathlib import Path

import pytest


from velocity_modelling.constants import get_data_root  # lazy resolver


def _env_path(var: str) -> Optional[Path]:
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

    # NOTE: do NOT resolve here; keep default None and resolve in a fixture.
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
        default=_env_path("NZVM_BINARY_PATH"),
        help="Path to nzvm binary (or set NZVM_BINARY_PATH env var).",
    )



@pytest.fixture(scope="session")
def benchmark_dir(pytestconfig: pytest.Config) -> Path:
    return Path(pytestconfig.getoption("--benchmark-dir")).expanduser().resolve()


@pytest.fixture(scope="session")
def data_root(pytestconfig: pytest.Config) -> Path:
    """
    Resolve NZCVM data root with precedence:
      1) --data-root (this option)
      2) NZCVM_DATA_ROOT env var
      3) ~/.config/nzcvm_data/config.json (written by `nzcvm-data install`)
      4) sensible defaults
      5) interactive prompt (disabled in tests)
    """
    cli_value: Optional[Path] = pytestconfig.getoption("--data-root")
    # Pass the CLI override as a string if provided; otherwise None
    resolved = get_data_root(str(cli_value) if cli_value else None)
    if not resolved.exists():
        raise FileNotFoundError(
            f"NZCVM data root not found: {resolved}. "
            "Run `nzcvm-data install`, or pass --data-root / set NZCVM_DATA_ROOT."
        )
    return resolved


@pytest.fixture(scope="session")
def nzvm_binary_path(pytestconfig: pytest.Config) -> Optional[Path]:
    p: Optional[Path] = pytestconfig.getoption("--nzvm-binary-path")
    return p.expanduser().resolve() if p else None

