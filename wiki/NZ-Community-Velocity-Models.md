# NZ Community Velocity Models

The NZ Community Velocity Models (NZCVM) provide detailed 3D representations of the subsurface velocity structure in New Zealand. These models are essential for seismic hazard analysis and earthquake engineering.

## Overview

The NZCVM incorporates various data sets amassed from numerous geophysical and geological studies. It prescribes Vp, Vs and Rho (density) at specified locations in a 3D grid.

The model consists of the following components:

-  **New Zealand-wide travel-time-derived seismic tomography model**: ~10km length scale
    - 2010 NZ: based on Eberhart-Philips et al. (2010)
    - 2020 NZ: based on Eberhart-Philips et al. (2020)

-  **Embedded subregion (sedimentary basin) models**: 34 basins of varying degrees of characterization

Through embedding discrete regional models into the lower resolution tomography data, we obtain a velocity model that incorporates data across multiple length scales and resolutions to give a unified representation of the velocity structure for use in broadband physics-based ground motion simulations and additional engineering applications.

## Key Components

- [**Basins**](Basins.md): Detailed information about the 34 basin models integrated into NZCVM
- [**Tomography**](Tomography.md): Information about available tomography models
- [**Data Formats**](DataFormats.md): Explanation of formats for surface, boundary, tomography, 1D velocity models, and smoothing data
- [**Output Formats**](OutputFormats.md): Details about the output formats including emod3d format

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

For detailed installation instructions, see the [Installation Guide](Installation.md).

## Using the `nzvm.py` Script

The `nzvm.py` script is a tool for generating velocity models based on the NZCVM. It allows users to specify configuration files and output directories to customize the generated models.

### Running the Script

To run the `nzvm.py` script, you need to provide specific arguments, including the configuration file and the output directory.

```sh
python cvm/scripts/nzvm.py generate-velocity-model /path/to/config/nzvm.cfg --out-dir /path/to/output
```

#### Example 1: Wellington 2p07

```sh
python cvm/scripts/nzvm.py generate-velocity-model tests/scenarios/Wellington_2p07/nzvm.cfg --out-dir OutDir/Wellington_2p07/Python
```

#### Example 2: Cant1D 2p07

```sh
python cvm/scripts/nzvm.py generate-velocity-model tests/scenarios/Cant1D_2p07/nzvm.cfg --out-dir OutDir/Cant1D_2p07/Python
```

### Configuration File: `nzvm.cfg`

The `nzvm.cfg` file is a configuration file used by the `nzvm.py` script to generate velocity models. It contains various parameters that define the properties and settings of the velocity model to be generated. Below is an explanation of the parameters in a sample `nzvm.cfg` file:

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

- **CALL_TYPE**: Specifies the type of operation to perform. In this case, it is set to `GENERATE_VELOCITY_MOD` to generate a velocity model.
- **MODEL_VERSION**: Indicates the version of the velocity model.
- **ORIGIN_LAT**: Latitude of the origin point for the model grid.
- **ORIGIN_LON**: Longitude of the origin point for the model grid.
- **ORIGIN_ROT**: Rotation angle of the model grid in degrees.
- **EXTENT_X**: Extent of the model grid in the X direction (in kilometers).
- **EXTENT_Y**: Extent of the model grid in the Y direction (in kilometers).
- **EXTENT_ZMAX**: Maximum depth of the model grid (in kilometers).
- **EXTENT_ZMIN**: Minimum depth of the model grid (in kilometers).
- **EXTENT_Z_SPACING**: Spacing between grid points in the Z direction (in kilometers).
- **EXTENT_LATLON_SPACING**: Spacing between grid points in the latitude and longitude directions (in degrees).
- **MIN_VS**: Minimum shear wave velocity (in meter per second).
- **TOPO_TYPE**: Type of topography to use. Possible values are `BULLDOZED`, `SQUASHED`, `SQUASHED_TAPERED` and `TRUE`. 
- **OUTPUT_DIR**: Directory where the generated velocity model files will be saved.

![TOPO_TYPE](images/topography_types.png)
*Figure 1: Different types of topography used in the NZCVM, including BULLDOZED, SQUASHED, SQUASHED_TAPERED, and TRUE.*

### Output Files

After successful execution, the output files will be located in the specified output directory. See [Output Formats](OutputFormats.md) for details on the format and contents of these files.

## References

Ethan M. Thomson, Brendon A. Bradley & Robin L. Lee (2020) Methodology and computational implementation of a New Zealand Velocity Model (NZVM2.0) for broadband ground motion simulation, New Zealand Journal of Geology and Geophysics, 63:1, 110-127, DOI: 10.1080/00288306.2019.1636830

Donna Eberhart-Phillips, Martin Reyners, Stephen Bannister, Mark Chadwick, Susan Ellis; Establishing a Versatile 3-D Seismic Velocity Model for New Zealand. Seismological Research Letters 2010; 81 (6): 992–1000. doi: https://doi.org/10.1785/gssrl.81.6.992

Donna Eberhart-Phillips, Stephen Bannister, Martin Reyners, and Stuart Henrys. "New Zealand Wide Model 2.2 Seismic Velocity and Qs and Qp Models for New Zealand". Zenodo, May 1, 2020. https://doi.org/10.5281/zenodo.3779523.


