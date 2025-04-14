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
cd velocity_modelling
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n velocity_modelling python=3.10
conda activate velocity_modelling
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The requirements include:
- Core scientific packages: numpy, h5py, pandas[parquet]
- Visualization: matplotlib
- Geospatial: shapely
- Testing: hypothesis[numpy], pytest, pytest-cov, pytest-repeat
- Project dependencies: numba, qcore, pyyaml, tqdm, typer
- Tool dependencies: cartopy, pytz, requests

### Step 4: Download Data (If Not Included in Repository)

Some data files may need to be downloaded separately due to their size:

```bash
# Example script to download data (if not provided)
python cvm/tools/download_data.py
```

Alternatively, download the file from Dropbox link. 
https://www.dropbox.com/scl/fi/53235uy9vmq8gdd58so4t/nzcvm_global_data.tar.gz?rlkey=0fpqa22fk6mf39iloe8s7lsa1&st=14xlz9or&dl=1


