# NZ Community Velocity Models

[![GitHub Actions](https://github.com/ucgmsim/velocity_modelling/workflows/CI/badge.svg)](https://github.com/ucgmsim/velocity_modelling/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

The NZ Community Velocity Models (NZCVM) provide detailed 3D representations of the subsurface velocity structure in New Zealand. These models are essential for seismic hazard analysis and earthquake engineering.

## Overview

The NZCVM incorporates various data sets amassed from numerous geophysical and geological studies. It prescribes the compression or primary wave velocity (Vp), the shear or secondary wave velocity (Vs) and density (Rho) at specified locations in a 3D grid.

The model consists of the following components:

-  **New Zealand-wide travel-time-derived seismic tomography model**: ~10km length scale
    - 2010 NZ: based on Eberhart-Philips et al. (2010)
    - 2020 NZ: based on Eberhart-Philips et al. (2020)

-  **Embedded subregion (sedimentary basin) models**: 34 basins of varying degrees of characterization

Through embedding discrete regional models into the lower resolution tomography data, we obtain a velocity model that incorporates data across multiple length scales and resolutions to give a unified representation of the velocity structure for use in broadband physics-based ground motion simulations and additional engineering applications.

## Key Components

- [**Basins**](wiki/Basins.md): Detailed information about the 34 basin models integrated into the NZCVM
- [**Tomography**](wiki/Tomography.md): Information about available tomography models
- [**Data Formats**](wiki/DataFormats.md): Explanation of formats for surface, boundary, tomography, 1D velocity models, and smoothing data
- [**Output Formats**](wiki/OutputFormats.md): Details about the output formats including emod3d format

## Requirements and Installation

### Prerequisites

- Python 3.8 or later
- Git (for cloning the repository)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ucgmsim/velocity_modelling.git
   cd velocity_modelling
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the data from the external link.

For data downloading and detailed installation instructions, see the [Installation Guide](wiki/Installation.md).

## Using the `nzcvm.py` Script

The `nzcvm.py` script is a tool for generating velocity models based on the NZCVM. It allows users to specify configuration files and output directories to customize the generated models.

### Running the Script

To run the `nzcvm.py` script, you need to provide specific arguments, including the configuration file and the output directory.

```sh
python cvm/scripts/nzcvm.py generate-velocity-model /path/to/config/nzcvm.cfg --out-dir /path/to/output
```


### Configuration File: `nzcvm.cfg`

The `nzcvm.cfg` file is a configuration file used by the `nzcvm.py` script to generate velocity models. It contains various parameters that define the properties and settings of the velocity model to be generated. Below is an explanation of the parameters in a sample `nzcvm.cfg` file:

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

For detailed instructions, see the [Generating Velocity Model](wiki/Generating-Velocity-Model.md) page.

### Output Files

After successful execution, the output files will be located in the specified output directory. See [Output Formats](wiki/OutputFormats.md) for details on the format and contents of these files.

## Data Files

All the data such as surface, boundary, DEM, 1D velocity model etc. files are placed under [velocity_modelling/cvm/data](velocity_modelling/data) folder. Details of the data format, see the [Data Formats Guide](wiki/DataFormats.md)

## Changelogs and Development Plans

For information about our development roadmap, upcoming features, and past version changes, please see our [Development Schedule](wiki/Development-Schedule.md).

## References

Ethan M. Thomson, Brendon A. Bradley & Robin L. Lee (2020) Methodology and computational implementation of a New Zealand Velocity Model (nzcvm2.0) for broadband ground motion simulation, New Zealand Journal of Geology and Geophysics, 63:1, 110-127, DOI: 10.1080/00288306.2019.1636830

Donna Eberhart-Phillips, Martin Reyners, Stephen Bannister, Mark Chadwick, Susan Ellis; Establishing a Versatile 3-D Seismic Velocity Model for New Zealand. Seismological Research Letters 2010; 81 (6): 992–1000. doi: https://doi.org/10.1785/gssrl.81.6.992

Donna Eberhart-Phillips, Stephen Bannister, Martin Reyners, and Stuart Henrys. "New Zealand Wide Model 2.2 Seismic Velocity and Qs and Qp Models for New Zealand". Zenodo, May 1, 2020. https://doi.org/10.5281/zenodo.3779523.
