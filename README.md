# NZ Community Velocity Model

[![GitHub Actions](https://github.com/ucgmsim/velocity_modelling/workflows/CI/badge.svg)](https://github.com/ucgmsim/velocity_modelling/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

The NZ Community Velocity Model (NZCVM) provide detailed 3D representations of New Zealand's subsurface velocity structure. These models are crucial for seismic hazard analysis and earthquake engineering.

## Overview

The NZCVM integrates various datasets from numerous geophysical and geological studies. It specifies the compression or primary wave velocity (Vp), the shear or secondary wave velocity (Vs), and density (Rho) at specified locations in a 3D grid.

The model includes the following components:

- **New Zealand-wide travel-time-derived seismic tomography model**: ~10 km length scale
    - 2010 NZ: based on Eberhart-Philips et al. (2010)
    - 2020 NZ: based on Eberhart-Philips et al. (2020)

- **Embedded subregion (sedimentary basin) models**: 41 basins with varying levels of characterization (version 2.08)

By embedding discrete regional models into the lower-resolution tomography data, the NZCVM provides a unified velocity model that incorporates data across multiple length scales and resolutions. This model is suitable for broadband physics-based ground motion simulations and other engineering applications.

## Key Components

- [**Basins**](wiki/Basins.md): Detailed information about the 41 basin models integrated into the NZCVM
- [**Tomography**](wiki/Tomography.md): Information about available tomography models
- [**Data Formats**](wiki/DataFormats.md): Explanation of formats for surface, boundary, tomography, 1D velocity models, and smoothing data
- [**Output Formats**](wiki/OutputFormats.md): Details about the output formats, including the emod3d, csv and HDF5 formats

## Requirements and Installation

### Prerequisites

- Python 3.11 or later
- Git (for cloning the repository)

### Installation

1. Clone the code repository:
     ```bash
     git clone https://github.com/ucgmsim/velocity_modelling.git
     cd velocity_modelling
     ```

2. Clone the data repository:
     ```bash
     git clone https://github.com/ucgmsim/nzcvm_data.git
     ```
     See [nzcvm_data installation instructions](https://github.com/ucgmsim/nzcvm_data#installation) for details.

3. Connect the data to the codebase:

    - **Option 1 (recommended):** Create a symbolic link:
      ```bash
      ln -s /path/to/nzcvm_data /path/to/velocity_modelling/velocity_modelling/nzcvm_data
      ```
      Replace `/path/to/` with your actual paths.

    - **Option 2:** Edit `DATA_ROOT` in `velocity_modelling/constants.py` to point to your data location:
      ```python
      # velocity_modelling/constants.py
      DATA_ROOT = "/absolute/path/to/nzcvm_data"
      ```

4. Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

For detailed installation and data setup instructions, see the [Installation Guide](wiki/Installation.md).



### Running the script

Currently, the NZCVM provides 3 python scripts located in the `scripts` directory. 
- `generate_3d_model.py`: Generates a 3D velocity model.
- `generate_1d_profiles.py`: Generates 1D velocity profiles.
- `extract_cross_section.py`: Extracts cross-sections from the velocity models.

#### Generating a 3D Velocity Model: `generate_3d_model.py`
To run the `generate_3d_model.py` script, provide specific arguments, including the configuration file and the output directory:

```sh
python scripts/generate_3d_model.py  /path/to/config/nzcvm.cfg --out-dir /path/to/output
```
##### Configuration File: `nzcvm.cfg`

The `nzcvm.cfg` file is a configuration file used by the `generate_3d_model.py` script to generate velocity models. It contains parameters defining the properties and settings of the velocity model. Below is an example:

```ini
CALL_TYPE=GENERATE_VELOCITY_MOD
MODEL_VERSION=2.03
ORIGIN_LAT=-43.4776
ORIGIN_LON=172.6870
ORIGIN_ROT=23.0
EXTENT_X=20
EXTENT_Y=20
EXTENT_ZMAX=45.0
EXTENT_ZMIN=0.0
EXTENT_Z_SPACING=0.2
EXTENT_LATLON_SPACING=0.2
MIN_VS=0.5
TOPO_TYPE=BULLDOZED
OUTPUT_DIR=/tmp
```

For detailed instructions, see the [Generating 3D Model](wiki/Generating-3D-Model.md) page.

##### Output Files

After successful execution, the output files will be located in the specified output directory. See [Output Formats](wiki/OutputFormats.md) for details on the format and contents of these files.



#### Generates 1D velocity profiles: `generate_1d_profiles.py`
To generate 1D velocity profiles, use the `generate_1d_profiles.py` script:

```sh
 python generate_1d_profiles.py --out-dir <output_directory> --model-version <version> --location-csv <csv_file> --min-vs <min_vs> --topo-type <topo_type> [--custom-depth <depth_file>] 
```
For detailed instructions, see the [Generating 1D Profiles](wiki/Generating-1D-Profiles.md) page.

#### Extracts cross-sections : `extract_cross_section.py`
To extract cross-sections from a HDF5-format velocity model, use the `extract_cross_section.py` script:

See [Extracting Cross-Sections](wiki/Extracting-Cross-Sections.md) for detailed instructions.



## Data Files

All data, such as surface, boundary, DEM, and 1D velocity model files, are located in the [velocity_modelling/data](velocity_modelling/data) folder. For details on the data format, see the [Data Formats Guide](wiki/DataFormats.md).

## Changelogs and Development Plans

For information about the development roadmap, upcoming features, and past version changes, see the [Development Schedule](wiki/Development-Schedule.md).

## References

- Ethan M. Thomson, Brendon A. Bradley & Robin L. Lee (2020). Methodology and computational implementation of a New Zealand Velocity Model (nzcvm2.0) for broadband ground motion simulation, *New Zealand Journal of Geology and Geophysics*, 63(1), 110-127. DOI: [10.1080/00288306.2019.1636830](https://doi.org/10.1080/00288306.2019.1636830)

- Donna Eberhart-Phillips, Martin Reyners, Stephen Bannister, Mark Chadwick, Susan Ellis (2010). Establishing a Versatile 3-D Seismic Velocity Model for New Zealand. *Seismological Research Letters*, 81(6), 992–1000. DOI: [10.1785/gssrl.81.6.992](https://doi.org/10.1785/gssrl.81.6.992)

- Donna Eberhart-Phillips, Stephen Bannister, Martin Reyners, and Stuart Henrys (2020). New Zealand Wide Model 2.2 Seismic Velocity and Qs and Qp Models for New Zealand. *Zenodo*. DOI: [10.5281/zenodo.3779523](https://doi.org/10.5281/zenodo.3779523)
