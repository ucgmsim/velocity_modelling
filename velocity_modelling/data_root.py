"""
Utilities to resolve the NZCVM data root directory.

"""

import json
import os
import sys
from pathlib import Path

# Mark the nzcvm-data runtime dependency as intentionally used (for deptry DEP002).
# We don't *need* the module API at runtime, but we ensure it's importable so deptry
# doesn't flag it as unused. Safe: only checks for presence.
try:
    import nzcvm_data as _nzcvm_data  # noqa: F401
except (ImportError, ModuleNotFoundError):
    _nzcvm_data = None

_NZCVM_DATA_RUNTIME_AVAILABLE = _nzcvm_data is not None



CONFIG_FILE = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    / "nzcvm_data"
    / "config.json"
)
DEFAULT_ROOT = Path(
    os.environ.get(
        "NZCVM_DATA_ROOT", Path.home() / ".local" / "cache" / "nzcvm_data_root"
    )
)


def _load_cfg_path() -> Path | None:
    """
    Load the data root path from the config file.

    Returns
    -------
    Path | None
        The data root path if found and valid, otherwise None.

    """
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:  # Use with open for file handling
                data = json.load(f)  # Use json.load to parse the JSON
            path_str = data.get("data_root")
            if not path_str:
                return None
            p = Path(path_str)
            return p if p.exists() else None
        except json.JSONDecodeError:
            print(
                f"Error: Invalid JSON format in {CONFIG_FILE}", file=sys.stderr
            )  # Specific error message
            return None
        except (
            TypeError,
            ValueError,
        ):  # Catch errors related to data type or value issues
            print(f"Error: Invalid data_root value in {CONFIG_FILE}", file=sys.stderr)
            return None
        except OSError as e:  # Catch file-related errors
            print(f"Error: OS error occurred: {e}", file=sys.stderr)
            return None
    return None


def _try_candidates() -> Path | None:
    """
    Try common candidate paths for the data root.

    Returns
    -------
    Path | None
        The first existing candidate path, otherwise None.

    """
    candidates = [
        os.environ.get("NZCVM_DATA_ROOT"),  # repeat in case changed at runtime
        DEFAULT_ROOT,
        Path.home() / "nzcvm_data",
        Path.cwd() / "nzcvm_data",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser().resolve()
        if p.exists():
            return p
    return None


def resolve_data_root(cli_override: str | None = None) -> Path:
    """
    Resolve NZCVM data root with precedence.

    Parameters
    ----------
    cli_override : str | None
        If provided, this path takes highest precedence.

    Returns
    -------
    Path
        Resolved data root path.

    """
    # 1) CLI override
    if cli_override:
        p = Path(cli_override).expanduser().resolve()
        if p.exists():
            return p

    # 2) env var
    env = os.environ.get("NZCVM_DATA_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    # 3) config written by `nzcvm-data install`
    p = _load_cfg_path()
    if p:
        return p

    # 4) sensible defaults / common locations
    p = _try_candidates()
    if p:
        return p

    # this is a hack, to make sure nzdvm_data is imported and keep depty happy (DEP002)
    if not _NZCVM_DATA_RUNTIME_AVAILABLE:
    # We won't fail here; just a hint. The CLI may still be on PATH if installed differently.
        pass

    raise FileNotFoundError(
        "Cannot locate NZCVM data root. "
        "Install or register the dataset with 'nzcvm-data install' "
        "or set --nzcvm-data-root / NZCVM_DATA_ROOT."
    )
