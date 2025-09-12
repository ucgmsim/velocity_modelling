# Installation and Usage Guide

This page provides detailed instructions for installing and using the NZCVM software.

## System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL recommended for best performance)
- **Python**: Python 3.11 or later
- **Disk Space**: Approximately 3Gb for the full dataset and code

## Installation

You can install directly from GitHub:

```bash
# Install the modelling code
pip install git+https://github.com/ucgmsim/velocity_modelling.git

# Fetch/update the data (no Python package needed)
nzcvm-data-helper ensure                 # clone or pull, no LFS
# or
nzcvm-data-helper ensure --full          # fetch large LFS files too

# Confirm where it is
nzcvm-data-helper where
# Optionally export for your shell
export NZCVM_DATA_ROOT="$(nzcvm-data-helper where)"
```



### Data Root Resolution

When locating the `nzcvm_data` repository, tools use this precedence:

1. `--nzcvm-data-root` CLI option
2. `NZCVM_DATA_ROOT` environment variable
3. `~/.config/nzcvm_data/config.json` (set by `nzcvm-data-helper ensure`)
4.  Default: `~/.local/cache/nzcvm_data_root`



### Create a Virtual Environment (Optional but Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n velocity_modelling python=3.10
conda activate velocity_modelling
```

### Development Dependencies
If you install from source, `requirements.txt` includes:
- Core scientific packages: `numpy`, `h5py`, `pandas[parquet]`
- Visualization: `matplotlib`
- Geospatial: `shapely`, `cartopy`
- Testing: `hypothesis[numpy]`, `pytest`, `pytest-cov`, `pytest-repeat`
- Project dependencies: `numba`, `qcore`, `pyyaml`, `tqdm`, `typer`
- Tool dependencies: `pytz`, `requests`

Installing with the -e (editable) option allows you to modify the source code locally and have changes reflected immediately—ideal for active development and keeping the software up to date.

