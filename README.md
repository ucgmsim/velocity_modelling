# NZ Community Velocity Model

[![GitHub Actions](https://github.com/ucgmsim/velocity_modelling/workflows/CI/badge.svg)](https://github.com/ucgmsim/velocity_modelling/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

This repository contains the **code** for generating and working with the NZ Community Velocity Model (NZCVM). The NZCVM provides detailed 3D representations of New Zealand's subsurface velocity structure for seismic hazard analysis and earthquake engineering.

**Note:** The actual data files are maintained in a separate repository: [nzcvm_data](https://github.com/ucgmsim/nzcvm_data)

## Overview

This codebase provides tools to:

- Generate 3D velocity models from various geophysical datasets
- Extract 1D velocity profiles at specific locations
- Create cross-sections from 3D velocity models
- Process and integrate multiple data sources (tomography, basin models, surface geology)

The NZCVM integrates datasets from numerous geophysical and geological studies, specifying compression wave velocity (Vp), shear wave velocity (Vs), and density (Rho) at specified locations in a 3D grid. By embedding discrete regional models into lower-resolution tomography data, the NZCVM provides a unified velocity model suitable for broadband physics-based ground motion simulations.

## Architecture

The system uses a **model version system** as the primary interface between code and data:

- **Code Repository** (this repo): Contains processing algorithms and model version definitions
- **Data Repository** ([nzcvm_data](https://github.com/ucgmsim/nzcvm_data)): Contains geophysical datasets and data registry
- **Model Versions**: YAML files that specify which data components to use and how to combine them

This separation allows independent version control of code and data while maintaining flexible model configurations.

## Key Components

- [**Model Versions**](wiki/Model-Versions.md): Primary interface defining which data components to use
- [**Output Formats**](wiki/OutputFormats.md): Details about the output formats, including the emod3d, csv and HDF5 formats

## Requirements and Installation

### Prerequisites

- Python 3.11 or later
- Git (for cloning the repository)

### Installation

#### **A. If you already installed `nzcvm_data`**

```bash
pip install git+https://github.com/ucgmsim/velocity_modelling.git
```

Your existing `NZCVM_DATA_ROOT` (set via nzcvm-data install) will be used automatically.
You can override at runtime with `--nzcvm-data-root /path/to/nzcvm_data` option.

#### **B. If you have not installed `nzcvm_data` yet**
When you install `velocity_modelling`, the lightweight `nzcvm-data` CLI is also installed.

Run it once to fetch/register the dataset:
```bash
pip install git+https://github.com/ucgmsim/velocity_modelling.git
nzcvm-data install  # full data (via git-lfs)
# or
nzcvm-data install --no-lfs  # lightweight data only
```


For detailed installation and data setup instructions, see the [Installation Guide](wiki/Installation.md).

## Command-Line Tools

The NZCVM provides 3 main command-line tools:

### 1. Generate 3D Models: `generate_3d_model`
Creates 3D velocity models from configuration files.

```bash
generate_3d_model /path/to/config/nzcvm.cfg --out-dir /path/to/output
```

Key features:
- Uses **model versions** to specify data components
- Supports multiple output formats (EMOD3D, CSV, HDF5)
- Configurable grid geometry and topography handling
- Web interface for configuration generation

For detailed instructions, see [Generating 3D Models](wiki/Generating-3D-Model.md).

### 2. Generate 1D Profiles: `generate_1d_profiles`
Extracts 1D velocity profiles at specific locations.

```bash
generate_1d_profiles --model-version 2.03 --location-csv sites.csv --out-dir /path/to/output
```

Key features:
- Uses same **model versions** as 3D generation
- Supports custom depth sampling
- Batch processing from CSV coordinates
- Multiple output formats

For detailed instructions, see [Generating 1D Profiles](wiki/Generating-1D-Profiles.md).

### 3. Extract Cross-sections: `extract_cross_section`
Creates cross-sections from existing 3D velocity models.

```bash
extract_cross_section /path/to/velocity_model.h5 --start-lat -41.0 --start-lon 174.0 --end-lat -41.5 --end-lon 175.0
```

For detailed instructions, see [Extracting Cross-Sections](wiki/Extracting-Cross-Sections.md).


## Model Version System

The **model version system** is the core interface that connects the velocity modeling code with the data repository. Model versions are YAML files (e.g., `2p03.yaml`, `2p07.yaml`) that specify:

- Which tomography models to use
- Which basin models to include  
- Surface handling methods
- Special processing options (offshore tapering, GTL, smoothing)

Example model version specification:
```ini
# In nzcvm.cfg
MODEL_VERSION=2.03
```

This automatically loads `model_versions/2p03.yaml` which defines the complete model configuration.

All tools (`generate_3d_model`, `generate_1d_profiles`, `extract_cross_section`) use this same model version system, ensuring consistency across different analysis workflows.

For complete details, see [Model Versions](wiki/Model-Versions.md).

## Data Files

All data files (surface models, boundaries, DEM, tomography models, etc.) are maintained in the separate [nzcvm_data](https://github.com/ucgmsim/nzcvm_data) repository. The data repository includes:

- **`nzcvm_registry.yaml`**: Catalog of all available datasets and their locations
- **Tomography models**: Regional and national velocity models
- **Basin models**: Detailed velocity structures for sedimentary basins
- **Surface data**: Topography, geology, and Vs30 datasets

See the [nzcvm_data README](https://github.com/ucgmsim/nzcvm_data) for details on available datasets and their formats.

For information about data format specifications used by this code, see the [Data Formats Guide](https://github.com/ucgmsim/nzcvm_data/blob/main/wiki/DataFormats.md).


## Quick Start Example

1. **Install** both repositories:
```bash
# Install code + data
pip install git+https://github.com/ucgmsim/velocity_modelling.git
nzcvm-data install
```

2. **Generate a 3D model**:
```bash
   # Create basic configuration
   cat > test.cfg << EOF
   CALL_TYPE=GENERATE_VELOCITY_MOD
   MODEL_VERSION=2.03
   ORIGIN_LAT=-43.5
   ORIGIN_LON=172.5
   ORIGIN_ROT=0.0
   EXTENT_X=10
   EXTENT_Y=10
   EXTENT_ZMAX=20.0
   EXTENT_ZMIN=0.0
   EXTENT_Z_SPACING=0.5
   EXTENT_LATLON_SPACING=0.1
   MIN_VS=0.5
   TOPO_TYPE=BULLDOZED
   OUTPUT_DIR=/tmp/test_model
   EOF
   
   # Generate model
   generate_3d_model test.cfg
```

3. **Extract 1D profiles**:
```bash
   # Create location file
   echo "lat,lon,name" > sites.csv
   echo "-43.5,172.5,site1" >> sites.csv
   
   # Generate profiles  
   generate_1d_profiles --model-version 2.03 --location-csv sites.csv --out-dir /tmp/profiles
```

## Changelogs and Development Plans

For information about the development roadmap, upcoming features, and past version changes, see the [Development Schedule](wiki/Development-Schedule.md).

## References

- Ethan M. Thomson, Brendon A. Bradley & Robin L. Lee (2020). Methodology and computational implementation of a New Zealand Velocity Model (nzcvm2.0) for broadband ground motion simulation, *New Zealand Journal of Geology and Geophysics*, 63(1), 110-127. DOI: [10.1080/00288306.2019.1636830](https://doi.org/10.1080/00288306.2019.1636830)

- Donna Eberhart-Phillips, Martin Reyners, Stephen Bannister, Mark Chadwick, Susan Ellis (2010). Establishing a Versatile 3-D Seismic Velocity Model for New Zealand. *Seismological Research Letters*, 81(6), 992–1000. DOI: [10.1785/gssrl.81.6.992](https://doi.org/10.1785/gssrl.81.6.992)

- Donna Eberhart-Phillips, Stephen Bannister, Martin Reyners, and Stuart Henrys (2020). New Zealand Wide Model 2.2 Seismic Velocity and Qs and Qp Models for New Zealand. *Zenodo*. DOI: [10.5281/zenodo.3779523](https://doi.org/10.5281/zenodo.3779523)
