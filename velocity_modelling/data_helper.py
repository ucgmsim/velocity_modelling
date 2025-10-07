"""
A small helper shipped with velocity_modelling to manage the *data* repo
without depending on a separate Python package.

Features:
  - `ensure`: clone if missing, otherwise `git pull` (optionally fetch LFS)
  - `where`: print the configured data root (config/env/defaults)

This writes ~/.config/nzcvm_data/config.json so velocity_modelling can
auto-discover the data root via get_data_root().

"""

import json
import os
import shutil
import subprocess
from json import JSONDecodeError
from pathlib import Path

import typer

from qcore import cli

APP_NAME = "nzcvm-data-helper"
DEFAULT_URL = "https://github.com/ucgmsim/nzcvm_data.git"
DEFAULT_ROOT = Path.home() / ".local" / "cache" / "nzcvm_data_root"
CONFIG_PATH = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    / "nzcvm_data"
    / "config.json"
)

app = typer.Typer(
    pretty_exceptions_enable=False,
    help="""
Helper to fetch/update the NZCVM data repository (no separate package required).

Behavior:
- Always re-aligns local repo to remote (fast-forward if possible, otherwise reset).
- Fetches large LFS files by default (full dataset). Use --no-full to skip LFS.

Examples:
  # Full dataset (default): clone if missing, else fetch/pull/reset, then fetch LFS
  nzcvm-data-helper ensure

  # Lightweight/CI: skip LFS (Only for testing or lightweight use cases. Not recommended for real use.)
  nzcvm-data-helper ensure --no-full

  # Use an already-cloned repo at a custom path, or specify remote/branch
  nzcvm-data-helper ensure --path /data/nzcvm_data
  nzcvm-data-helper ensure --repo https://github.com/ucgmsim/nzcvm_data.git --branch develop

  # Print the currently configured data root
  nzcvm-data-helper where
""",
)


def _run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> None:
    """
    Run a command, raising CalledProcessError on failure.

    Parameters
    ----------
    cmd : list[str]
        Command and arguments to run.
    cwd : Path | None
        Working directory to run the command in.
    check : bool
        If True, raise CalledProcessError on non-zero exit. Default is True.

    Raises
    ------
    subprocess.CalledProcessError
        If the command fails and check=True.

    """
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def _require_bin(name: str, msg: str | None = None):
    """
    Ensure a command-line binary is available.

    Parameters
    ----------
    name : str
        Name of the binary to check.
    msg : str | None
        Custom error message if not found.

    Raises
    ------
    typer.Exit
        If the binary is not found.
    """
    if not shutil.which(name):
        typer.echo(msg or f"[{APP_NAME}] required command not found: {name}", err=True)
        raise typer.Exit(code=1)


def _write_config(path: Path):
    """
    Write the config file pointing to the given data root.

    Parameters
    ----------
    path : Path
        The data root path to write to the config.
    """
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"data_root": str(path)}
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def _looks_like_repo(path: Path) -> bool:
    """
    Heuristic check if the given path looks like a valid repo.

    Parameters
    ----------
    path : Path
        The path to check.

    Returns
    -------
    bool
        True if the path looks like a valid repo, False otherwise.
    """
    # Heuristic: must be a git repo and contain the registry file
    return (path / ".git").exists() and (path / "nzcvm_registry.yaml").exists()


def _detect_upstream_branch(repo_path: Path) -> str:
    """
    Detect the current upstream branch name (e.g., origin/main); fallback to 'main'

    Parameters
    ----------
    repo_path : Path
        The path to the git repository.

    Returns
    -------
    str
        The name of the upstream branch, or 'main' if detection fails.
    """
    try:
        out = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_path),
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{u}",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out and "/" in out:
            return out.split("/", 1)[1]  # e.g., origin/main -> main
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        pass
    return "main"


@cli.from_docstring(app)
def ensure(
    path: Path = DEFAULT_ROOT,
    repo: str = DEFAULT_URL,
    branch: str | None = None,
    full: bool = True,
    write_config: bool = True,
    quiet: bool = False,
):
    """
    Clone if missing; otherwise pull. Optionally fetch LFS files with --full.
    This writes ~/.config/nzcvm_data/config.json so velocity_modelling can
    auto-discover the data root via get_data_root().

    Parameters
    ----------
    path : Path, optional
        Where to place/update the local clone.
    repo : str, optional
        Git URL for the data repository. Default is https://github.com/ucgmsim/nzcvm_data.git
    branch : str | None , optional
        Branch to checkout (clone or existing). Default is None.

    full : bool, optional
            Fetch large files via git-lfs (HDF5, etc.). Requires git-lfs to be installed. Default is True.
    write_config : bool, optional
        If True, (over)write ~/.config/nzcvm_data/config.json. for auto-discovery. Default is True.
    quiet : bool
        Reduce logging.

    Raises
    ------
    typer.Exit
        On various errors (git not found, git commands fail, unexpected layout, etc.)
    """
    _require_bin("git")

    # If full dataset requested, require git-lfs *before* doing any clone/pull work.

    if full:
        _require_bin(
            "git-lfs",
            "git-lfs is required for full dataset. Install git-lfs (See https://git-lfs.com) and try again.",
        )

    if not quiet:
        typer.echo(f"[{APP_NAME}] target: {path}")

    if path.exists() and _looks_like_repo(path):
        if not quiet:
            typer.echo(f"[{APP_NAME}] updating existing repo: {path}")
        target_branch = branch or _detect_upstream_branch(path)
        # Ensure on the desired branch
        if branch:
            try:
                _run(["git", "-C", str(path), "checkout", branch])
            except subprocess.CalledProcessError:
                typer.echo(f"[{APP_NAME}] git checkout {branch} failed", err=True)
                raise typer.Exit(code=1)

        # Fetch latest
        try:
            _run(["git", "-C", str(path), "fetch", "--prune"])
        except subprocess.CalledProcessError:
            typer.echo(f"[{APP_NAME}] git fetch failed", err=True)
            raise typer.Exit(code=1)

        try:
            _run(["git", "-C", str(path), "pull", "--ff-only"])
        except subprocess.CalledProcessError:
            typer.echo(
                f"[{APP_NAME}] pull not possible. Resetting to origin/{target_branch} ..."
            )
            try:
                _run(
                    [
                        "git",
                        "-C",
                        str(path),
                        "reset",
                        "--hard",
                        f"origin/{target_branch}",
                    ]
                )
            except subprocess.CalledProcessError:
                typer.echo(
                    f"[{APP_NAME}] git reset --hard origin/{target_branch} failed",
                    err=True,
                )
                raise typer.Exit(code=1)

    else:
        # (Re)clone
        path.parent.mkdir(parents=True, exist_ok=True)
        if not quiet:
            typer.echo(f"[{APP_NAME}] cloning {repo} -> {path}")
        clone_cmd = ["git", "clone", "--filter=blob:none"]
        if branch:
            clone_cmd += ["-b", branch]
        clone_cmd += [repo, str(path)]
        try:
            _run(clone_cmd)
        except subprocess.CalledProcessError:
            typer.echo(f"[{APP_NAME}] git clone failed", err=True)
            raise typer.Exit(code=1)

    # Optionally fetch LFS assets (default = full)
    if full:
        # Install LFS hooks in this environment (safe if repeated)
        _run(["git", "-C", str(path), "lfs", "install"])
        if not quiet:
            typer.echo(f"[{APP_NAME}] fetching LFS objects...")
        _run(["git", "-C", str(path), "lfs", "pull"])

    # Verify a key file exists
    if not (path / "nzcvm_registry.yaml").exists():
        typer.echo(
            f"[{APP_NAME}] unexpected layout: nzcvm_registry.yaml not found at {path}",
            err=True,
        )
        raise typer.Exit(code=2)

    # Persist config for auto-discovery
    if write_config:
        _write_config(path)
        if not quiet:
            typer.echo(f"[{APP_NAME}] wrote config: {CONFIG_PATH}")

    # Friendly hint for env usage
    if not quiet:
        typer.echo(f"[{APP_NAME}] NZCVM data root ready: {path}")
        typer.echo(f'export NZCVM_DATA_ROOT="{path}"')


@cli.from_docstring(app)
def where(
    print_export: bool = False,
):
    """
    Print the currently configured data root, if any.

    Parameters
    ----------
    print_export : bool, optional
        If True, print as shell export line: export NZCVM_DATA_ROOT="...". Default is False.

    """
    # 1) config file
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            root = data.get("data_root")
            if root:
                out = f'export NZCVM_DATA_ROOT="{root}"' if print_export else root
                typer.echo(out)
                raise typer.Exit()
        except (OSError, JSONDecodeError):
            pass
    # 2) env var
    env = os.environ.get("NZCVM_DATA_ROOT")
    if env:
        out = f'export NZCVM_DATA_ROOT="{env}"' if print_export else env
        typer.echo(out)
        raise typer.Exit()
    # 3) sensible default if it exists
    if DEFAULT_ROOT.exists():
        out = (
            f'export NZCVM_DATA_ROOT="{DEFAULT_ROOT}"'
            if print_export
            else str(DEFAULT_ROOT)
        )
        typer.echo(out)
        raise typer.Exit()
    typer.echo("(not configured)", err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
