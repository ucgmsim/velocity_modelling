# Installation and Usage Guide

This page provides detailed instructions for installing and using the NZCVM software.

## System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL recommended for best performance)
- **Python**: Python 3.11 or later
- **Disk Space**: Approximately 3Gb for the full dataset and code

## Installation

### Step 1: Install the Code

You can install directly from GitHub:

```bash
pip install git+https://github.com/ucgmsim/velocity_modelling.git
```
This will also install the lightweight[] [`nzcvm-data`](https://github.com/ucgmsim/nzcvm_data)
 package, which manages the dataset.

If you are developing the codebase:
```bash
git clone https://github.com/ucgmsim/velocity_modelling.git
cd velocity_modelling
pip install -r requirements.txt && pip install -e .
```

### Step 2: Install the Data (via the CLI)
The actual model data are hosted in the separate [`nzcvm_data`](https://github.com/ucgmsim/nzcvm_data)repository.
You do *not* need to clone it manually. Instead, use the `nzcvm-data` CLI to set it up:
```bash
# Full dataset (requires git-lfs, includes large HDF5 files)
nzcvm-data install

# Or, lightweight mode (skips LFS; only small files and boundaries)
nzcvm-data install --no-lfs
```
The CLI clones the data repository into a cache location (default: ~/.local/cache/nzcvm_data_root) and writes a config file at ~/.config/nzcvm_data/config.json.

You can check the configured location:
```bash
nzcvm-data where
```
If you already have a clone, you can register it instead:
```bash
nzdvm-data install --path /path/to/existing/nzcvm_data
```


### Data Root Resolution

When locating the `nzcvm_data` repository, tools use this precedence:

1. `--nzcvm-data-root` CLI option
2. `NZCVM_DATA_ROOT` environment variable
3. `~/.config/nzcvm_data/config.json` (set by `nzcvm-data install`)
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

