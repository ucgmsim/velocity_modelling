"""
Utilities to resolve the NZCVM data root directory.

"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

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
            path_str = data.get("data_root", "")
            p = Path(path_str)
            return p if p.exists() else None
        except FileNotFoundError:
            print(
                f"Error: Config file not found at {CONFIG_FILE}"
            )  # Specific error message
            return None
        except json.JSONDecodeError:
            print(
                f"Error: Invalid JSON format in {CONFIG_FILE}"
            )  # Specific error message
            return None
        except (
            TypeError,
            ValueError,
        ):  # Catch errors related to data type or value issues
            print(f"Error: Invalid data_root value in {CONFIG_FILE}")
            return None
        except OSError as e:  # Catch file-related errors
            print(f"Error: OS error occurred: {e}")
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


def resolve_data_root(
    cli_override: str | None = None, interactive: bool | None = None
) -> Path:
    """
    Resolve NZCVM data root with precedence.

    Parameters
    ----------
    cli_override : str | None
        If provided, this path takes highest precedence.
    interactive : bool | None
        If None (default), prompt if stdin/stdout are TTY. If True/False

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

    # 5) interactive prompt (if TTY)
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if interactive:
        entered = input(
            "Enter NZCVM_DATA_ROOT (or leave blank to run 'nzcvm-data install'): "
        ).strip()
        if entered:
            p = Path(entered).expanduser().resolve()
            if p.exists():
                return p
            print(f"Path not found: {p}", file=sys.stderr)
        exe = shutil.which("nzcvm-data")
        if exe:
            print("Running 'nzcvm-data install' to set up data...")
            subprocess.run([exe, "install"], check=False)
            p2 = _load_cfg_path()
            if p2 and p2.exists():
                return p2

    raise FileNotFoundError(
        "Cannot locate NZCVM data root. "
        "Install or register the dataset with 'nzcvm-data install' "
        "or set --nzcvm-data-root / NZCVM_DATA_ROOT."
    )
