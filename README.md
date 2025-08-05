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

For detailed data downloading and installation instructions, see the [Installation Guide](wiki/Installation.md).



### Running the script

Currently, the NZCVM provides 3 Python scripts  located in the `scripts` directory. 
- `generate_3d_model.py`: Generates velocity models based on the configuration files.
- `generate_1d_profiles.py`: Generates 1D velocity profiles.
- `extract_cross_section.py`: Extracts cross-sections from the velocity models.

#### `generate_3d_model.py`
To run the `generate_3d_model.py` script, provide specific arguments, including the configuration file and the output directory:

```sh
python scripts/generate_3d_model.py  /path/to/config/nzcvm.cfg --out-dir /path/to/output
```
#### `generate_1d_profiles.py`
To generate 1D velocity profiles, use the `generate_1d_profiles.py` script:

```sh
 python generate_1d_profiles.py --out-dir <output_directory> --model-version <version> --location-csv <csv_file> --min-vs <min_vs> --topo-type <topo_type> [--custom-depth <depth_file>] 
```

#### `extract_cross_section.py`
To extract cross-sections from a HDF5-format velocity model, use the `extract_cross_section.py` script:

```sh
    python extract_cross_section.py <h5file> [options]
    --lat1 <lat1> --lon1 <lon1> --lat2 <lat2> --lon2 <lon2>
    --x1 <x1> --y1 <y1> --x2 <x2> --y2 <y2>
    --property <property_name> --xaxis <xaxis> --n_points <n_points>
    --vmin <vmin> --vmax <vmax> --max_depth <max_depth>
    --cmap <cmap> --png <output_png> --csv <csv_output>

```
### Configuration File: `nzcvm.cfg`

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

For detailed instructions, see the [Generating Velocity Model](wiki/Generating-Velocity-Model.md) page.

### Output Files

After successful execution, the output files will be located in the specified output directory. See [Output Formats](wiki/OutputFormats.md) for details on the format and contents of these files.

## Data Files

All data, such as surface, boundary, DEM, and 1D velocity model files, are located in the [velocity_modelling/data](velocity_modelling/data) folder. For details on the data format, see the [Data Formats Guide](wiki/DataFormats.md).

## Changelogs and Development Plans

For information about the development roadmap, upcoming features, and past version changes, see the [Development Schedule](wiki/Development-Schedule.md).

## References

- Ethan M. Thomson, Brendon A. Bradley & Robin L. Lee (2020). Methodology and computational implementation of a New Zealand Velocity Model (nzcvm2.0) for broadband ground motion simulation, *New Zealand Journal of Geology and Geophysics*, 63(1), 110-127. DOI: [10.1080/00288306.2019.1636830](https://doi.org/10.1080/00288306.2019.1636830)

- Donna Eberhart-Phillips, Martin Reyners, Stephen Bannister, Mark Chadwick, Susan Ellis (2010). Establishing a Versatile 3-D Seismic Velocity Model for New Zealand. *Seismological Research Letters*, 81(6), 992–1000. DOI: [10.1785/gssrl.81.6.992](https://doi.org/10.1785/gssrl.81.6.992)

- Donna Eberhart-Phillips, Stephen Bannister, Martin Reyners, and Stuart Henrys (2020). New Zealand Wide Model 2.2 Seismic Velocity and Qs and Qp Models for New Zealand. *Zenodo*. DOI: [10.5281/zenodo.3779523](https://doi.org/10.5281/zenodo.3779523)
