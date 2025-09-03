# Installation and Usage Guide

This page provides detailed instructions for installing and using the NZCVM software.

## System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL recommended for best performance)
- **Python**: Python 3.11 or later
- **Disk Space**: Approximately 10GB for the full dataset and code

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/ucgmsim/velocity_modelling.git
```

### Step 2: Clone the Data Repository

The NZCVM data files are now maintained in a separate repository: [`nzcvm_data`](https://github.com/ucgmsim/nzcvm_data).

Clone the data repository:

```bash
git clone https://github.com/ucgmsim/nzcvm_data.git
```

Follow any additional instructions in the [`nzcvm_data` README](https://github.com/ucgmsim/nzcvm_data#installation) if required.

### Step 3: Connect the Data to the Code

You have two options to make the data available to the NZCVM code:

**Option 1: Create a symbolic link (recommended)**

Create a symbolic link from the cloned `nzcvm_data` directory to the expected location inside the codebase:

```bash
ln -s /path/to/nzcvm_data /path/to/velocity_modelling/velocity_modelling/nzcvm_data
```

Replace `/path/to/` with your actual paths.

**Option 2: Update DATA_ROOT in constants.py**

Alternatively, you can edit the `DATA_ROOT` variable in `velocity_modelling/constants.py` to point directly to your `nzcvm_data` location:

```python
# velocity_modelling/constants.py
DATA_ROOT = "/absolute/path/to/nzcvm_data"
```

This is useful if you do not wish to use a symbolic link.

### Step 4: Create a Virtual Environment (Optional but Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n velocity_modelling python=3.10
conda activate velocity_modelling
```

### Step 5: Install

```bash
cd velocity_modelling
pip install -r requirements.txt && pip install -e .
```

The requirements include:
- Core scientific packages: numpy, h5py, pandas[parquet]
- Visualization: matplotlib
- Geospatial: shapely
- Testing: hypothesis[numpy], pytest, pytest-cov, pytest-repeat
- Project dependencies: numba, qcore, pyyaml, tqdm, typer
- Tool dependencies: cartopy, pytz, requests

Installing with the -e (editable) option allows you to modify the source code locally and have changes reflected immediately—ideal for active development and keeping the software up to date.

