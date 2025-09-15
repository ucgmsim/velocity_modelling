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


### Install and Ensure the Data

Install the NZCVM modeling code, which includes the `nzcvm-data-helper` CLI tool:
```bash
# Install the modelling codes
pip install git+https://github.com/ucgmsim/velocity_modelling.git
```

Then, ensure you have the data repository. If you don't have it yet, the helper will clone it for you. 
If you already have it, the helper will pull the latest changes. 
If you have it in a custom location, you can specify that too.

```bash
# Full dataset (default, includes LFS)
nzcvm-data-helper ensure

# For lightweight CI/test installs (skip LFS, *Not* recommended for production):
nzcvm-data-helper ensure --no-full

# Use a custom path if you already cloned nzcvm_data:
nzcvm-data-helper ensure --path /path/to/nzcvm_data
```

Check where the data is located:

```bash
# Confirm where it is
nzcvm-data-helper where
```
Then, you can optionally export the path for your shell:
```bash
# Optionally export for your shell
export NZCVM_DATA_ROOT="$(nzcvm-data-helper where)"
```

### Notes

- By default, `nzcvm-data-helper ensure` installs the **full dataset**, including large LFS files (multi-GB).

- Use `--no-full `to skip LFS for lightweight test. This is useful for CI or if you only need metadata.

- The helper always re-aligns your local clone to the remote (reset if necessary).

### Data Root Resolution

When locating the `nzcvm_data` repository, tools use this precedence:

1. `--nzcvm-data-root` CLI option
2. `NZCVM_DATA_ROOT` environment variable
3. `~/.config/nzcvm_data/config.json` (set by `nzcvm-data-helper ensure`)
4.  Default: `~/.local/cache/nzcvm_data_root`


### Development Dependencies
If you install from source, `requirements.txt` includes:
- Core scientific packages: `numpy`, `h5py`, `pandas[parquet]`
- Visualization: `matplotlib`
- Geospatial: `shapely`, `cartopy`
- Testing: `hypothesis[numpy]`, `pytest`, `pytest-cov`, `pytest-repeat`
- Project dependencies: `numba`, `qcore`, `pyyaml`, `tqdm`, `typer`
- Tool dependencies: `pytz`, `requests`


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