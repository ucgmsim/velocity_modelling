# Installation and Usage Guide

This page provides detailed instructions for installing and using the NZCVM software.

## System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL recommended for best performance)
- **Python**: Python 3.11 or later
- **Disk Space**: Approximately 3Gb for the full dataset and code

## Installation

### Create a Virtual Environment (Optional but Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n velocity_modelling python=3.11
conda activate velocity_modelling
```


### Step 1: Install the Code

You can install directly from GitHub:

```bash
pip install git+https://github.com/ucgmsim/velocity_modelling.git
```
This will also install the lightweight [`nzcvm-data`](https://github.com/ucgmsim/nzcvm_data)
 package, which manages the dataset.

If you are developing the codebase:
```bash
git clone https://github.com/ucgmsim/velocity_modelling.git
cd velocity_modelling
pip install -r requirements.txt && pip install -e .
```

### Step 2: Install the Data (via the CLI)
The actual model data are hosted in the separate [`nzcvm_data`](https://github.com/ucgmsim/nzcvm_data) repository.
You do *not* need to clone it manually. Instead, use the `nzcvm-data` CLI to set it up:
```bash
# Full dataset (requires git-lfs, includes large HDF5 files)
nzcvm-data install

# Or, lightweight mode (skips LFS; only small files and boundaries) 
# - useful for testing, but cannot generate full models
nzcvm-data install --no-lfs
```
The CLI clones the data repository into a cache location (default: ~/.local/cache/nzcvm_data_root) and writes a config file at ~/.config/nzcvm_data/config.json.

You can check the configured location:
```bash
nzcvm-data where
```
If you already have a clone, you can register it instead:
```bash
nzcvm-data install --path /path/to/existing/nzcvm_data
```


### Data Root Resolution

When locating the `nzcvm_data` repository, tools use this precedence:

1. `--nzcvm-data-root` CLI option
2. `NZCVM_DATA_ROOT` environment variable
3. `~/.config/nzcvm_data/config.json` (set by `nzcvm-data install`)
4.  Default: `~/.local/cache/nzcvm_data_root`


### Development Dependencies
If you install from source, `requirements.txt` includes:
- Core scientific packages: `numpy`, `h5py`, `pandas[parquet]`
- Visualization: `matplotlib`
- Geospatial: `shapely`, `cartopy`
- Testing: `hypothesis[numpy]`, `pytest`, `pytest-cov`, `pytest-repeat`
- Project dependencies: `numba`, `qcore`, `pyyaml`, `tqdm`, `typer`
- Tool dependencies: `pytz`, `requests`

Installing with the -e (editable) option allows you to modify the source code locally and have changes reflected immediately—ideal for active development and keeping the software up to date.

## Install and Run with Docker

You can run the tools inside a container to avoid host setup differences.

### Option A — Use your existing data (recommended)

Mount your existing NZCVM data root (created by `nzcvm-data install` on the host) into the container.

**Dockerfile (example):**
```dockerfile
FROM python:3.11-slim

# System deps for scientific stack & git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential gfortran ca-certificates \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

# Install velocity_modelling (pulls the nzcvm-data CLI)
RUN pip install --no-cache-dir git+https://github.com/ucgmsim/velocity_modelling.git

# Path where we'll mount the dataset
ENV NZCVM_DATA_ROOT=/opt/nzcvm_data

WORKDIR /work
```

**Build the image:**
```bash
docker build -t nzcvm:latest .
```

**Run (mount your working dir and the host data root):**
```bash
# Replace the right-hand side of the second -v with your host data root path
docker run --rm -it \
  -v "$PWD":/work \
  -v "$HOME/.local/cache/nzcvm_data_root":/opt/nzcvm_data:ro \
  -w /work \
  nzcvm:latest \
  generate_3d_model path/to/nzcvm.cfg --out-dir out/
```

Notes:
- The data volume is mounted **read-only** (`:ro`) for safety.
- You can override the location with `--nzcvm-data-root /opt/nzcvm_data` if desired, but the `ENV` above already points there.

### Option B — Install data inside the container

You can also fetch data in the container (useful for CI). This makes the image heavier.

Add to the Dockerfile **after** installing the package:
```dockerfile
# (Optional) Install full dataset inside the image
# Remove --no-lfs if you want the full HDF5s
RUN nzcvm-data install --no-lfs \
 && nzcvm-data where
```

Then run the container *without* a data mount:
```bash
docker run --rm -it -v "$PWD":/work -w /work nzcvm:latest   generate_3d_model path/to/nzcvm.cfg --out-dir out/
```

**Tip for CI**: you can toggle a full install vs. light install using build args or environment variables (e.g., `--no-lfs`).

### Passing config & writing outputs

Make sure to mount the directory containing your config and a place to write outputs:
```bash
docker run --rm -it \
  -v "/abs/path/to/configs":/configs \
  -v "/abs/path/to/outputs":/outputs \
  -v "$HOME/.local/cache/nzcvm_data_root":/opt/nzcvm_data:ro \
  -w /configs \
  nzcvm:latest \
  generate_3d_model nzcvm.cfg --out-dir /outputs
```