# NZ Community Velocity Models

The NZ Community Velocity Models (NZCVM) provide detailed 3D representations of the subsurface velocity structure in New Zealand. These models are essential for seismic hazard analysis and earthquake engineering.

## Overview

The NZCVM incorporates various data sets amassed from numerous geophysical and geological studies. 
- Multiple datasets of New Zealand-wide travel-time-derived tomography model:
    - 2010 NZ : based on Eberhart-Philips et al. (2010)
    - 2020 NZ : based on Eberhart-Philips et al. (2020)

- Regional data incorporating 34 basins of varying degrees of characterization:
  - [Canterbury Pre Quaternary](basins/Canterbury_Pre_Quaternary.md)
  - [Canterbury North](basins/Canterbury_North.md)
  - [Banks Peninsula Volcanics](basins/Banks_Peninsula_Volcanics.md)
  - [Kaikoura](basins/Kaikoura.md)
  - [Cheviot](basins/Cheviot.md)
  - [Hanmer](basins/Hanmer.md)
  - [Marlborough](basins/Marlborough.md)
  - [Nelson](basins/Nelson.md)
  - [Wellington](basins/Wellington.md)
  - [Waikato Hauraki](basins/WaikatoHauraki.md)
  - [Wanaka](basins/Wanaka.md)
  - [MacKenzie](basins/MacKenzie.md)
  - [Wakatipu](basins/Wakatipu.md)
  - [Alexandra](basins/Alexandra.md)
  - [Ranfurly](basins/Ranfurly.md)
  - [NE Otago](basins/NE_Otago.md)
  - [Mosgiel](basins/Mosgiel.md)
  - [Balclutha](basins/Balclutha.md)
  - [Dunedin](basins/Dunedin.md)
  - [Murchison](basins/Murchison.md)
  - [Waitaki](basins/Waitaki.md)
  - [Hakataramea](basins/Hakataramea.md)
  - [Karamea](basins/Karamea.md)
  - [Collingwood Basin](basins/CollingwoodBasin.md)
  - [Springs Junction](basins/SpringsJunction.md)
  - [Hawkes Bay](basins/HawkesBay.md)
  - [Napier](basins/Napier.md)
  - [Greater Wellington](basins/GreaterWellington.md)
  - [Porirua](basins/Porirua.md)
  - [Gisborne](basins/Gisborne.md)
  - [Southern Hawkes Bay](basins/SouthernHawkesBay.md)
  - [Wairarapa](basins/Wairarapa.md)
  - [Motu Bay](basins/Motu_Bay.md)
  - [Whangaparoa](basins/Whangaparoa.md)

![nzvm_2p07_basin_map](https://github.com/user-attachments/assets/ef94b323-fb08-4f39-8666-ffaf12ae4db4)

Through embedding discrete regional models into the lower resolution tomography data, we obtain a velocity model that incorporates data across multiple length scales and resolutions to give a unified representation of the velocity structure for use in broadband physics-based ground motion simulations and additional engineering applications. 

## Using the `nzvm.py` Script

The `nzvm.py` script is a tool for generating velocity models based on the NZCVM. It allows users to specify configuration files and output directories to customize the generated models.

### Prerequisites

- Python 3.12 or later
- Required Python packages (install via `requirements.txt`)

### Running the Script

To run the `nzvm.py` script, you need to provide specific arguments, including the configuration file and the output directory. Below are examples of how to run the script with different configurations.

#### Example 1: Wellington 2p07

```sh
/path/to/python /path/to/velocity_modelling/cvm/scripts/nzvm.py generate-velocity-model /path/to/velocity_modelling/tests/scenarios/Wellington_2p07/nzvm.cfg --out-dir /path/to/velocity_modelling/OutDir/Wellington_2p07/Python
```

#### Example 2: Cant1D 2p07

```sh
/path/to/python /path/to/velocity_modelling/cvm/scripts/nzvm.py generate-velocity-model /path/to/velocity_modelling/tests/scenarios/Cant1D_2p07/nzvm.cfg --out-dir /path/to/velocity_modelling/OutDir/Cant1D_2p07/Python
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
- **TOPO_TYPE**: Type of topography to use. Possible values are `BULLDOZED`, `SQUASHED`,`SQUASHED_TAPERED` and `TRUE`. 
- **OUTPUT_DIR**: Directory where the generated velocity model files will be saved.

### Model Versions

The `model_versions` section defines different versions of the velocity models. Each version is specified in a YAML file that contains detailed parameters and settings for that version. Below is an example of a `2p03.yaml` file:

```yaml
GTL: true
surfaces:
  - name: NZ_DEM
    submodel: ep_tomography_submod_v2010

tomography: 2010_NZ_OFFSHORE
basin_edge_smoothing: true
basins:
  - Canterbury_Pre_Quaternary_v19p1
  - Canterbury_North_v19p1
  - Banks_Peninsula_Volcanics_v19p1
  - Kaikoura_v19p1
  - Cheviot_v19p1
  - Hanmer_v19p1
  - Marlborough_v19p1
  - Nelson_v19p1
  - Wellington_v19p6
  - WaikatoHauraki_v19p7
```

- **GTL**: Indicates whether the Geotechnical Layer (GTL) is included (`true` or `false`).
- **surfaces**: Lists the surfaces used in the model. Each surface has a `name` and an associated `submodel`.
  - **name**: The name of the surface.
  - **submodel**: The submodel associated with the surface. The submodel defines how to compute/assign vp,vs, and rho values
- **tomography**: Specifies the tomography model used.
- **basin_edge_smoothing**: Indicates whether basin edge smoothing is applied (`true` or `false`).
- **basins**: Lists the basins included in the model. Each basin is identified by its name.

User can create a custom model version by placing a .yaml file in `model_version` folder. Our convention is to use `p` in place of `.`.  Edit the `MODEL_VERION` field in `nzvm.cfg` or `--model-version` argument when executing `nzvm` command.

## NZVM Registry
`nzvm_registry.yaml` contains the details of surface, tomography, basin and submodel data.
Let us explore the registry using the `2p03.yaml` example.

The "surfaces" for `model_version 2.03` was defined as 

```
surfaces:
  - name: NZ_DEM
    submodel: ep_tomography_submod_v2010
```
`NZ_DEM` lives in the registry as:

```
surface:
  - name: NZ_DEM
    path: DEM/NZ_DEM_HD.in
```

The associated submodel `ep_tomography_submod_v2010` is defined as:

```
submodel:
...
    - name: ep_tomography_submod_v2010
      type: tomography
...      
```
As this is of `tomography` type, it will partner with the value for "tomography" (ie. `tomography: 2010_NZ_OFFSHORE`)

`2010_NZ_OFFSHORE` is defined as 
```
tomography:
...
  - name: 2010_NZ_OFFSHORE
    nElev: 20
    elev: [ 15, 1, -3, -8, -15, -23, -30, -38, -48, -65, -85, -105, -130, -155, -185, -225, -275, -370, -620, -750 ]
    vs30_path: Global_Surfaces/NZ_Vs30_HD_With_Offshore.in
    special_offshore_tapering: true
    path: Tomography/2010_NZ
    format: HDF5
```
The format of `tomography` data is given in [Tomography](../format/Tomography.md).

An example of basin model is:

```
basin:
...
  - name: Cheviot_v19p1
    boundaries:
      - SI_BASINS/Cheviot_Polygon_WGS84.txt
    surfaces:
      - name: NZ_DEM
        submodel: canterbury1d_v2

      - name: CheviotBasement

    smoothing: Boundaries/Smoothing/Cheviot_v19p1.txt
```
Each basin needs 4 major items to define.

boundary: a close-loop (ie. polygon)) of (lat, lon) coordinates (can have multiple boundary files)
surface: depth at a grid point. Typically, the top surface is NZ_DEM (top-level surface), and the subsequent surfaces are the identified layers.
submomdel: Each surface has an associated submodel that defines how to assign/compute vp,vs,rho for this surface. The above means we use canterbury1d_v2 (renamed from the widely used Cant1D_v2) for the layer between NZ_DEM and CheviotBasement. As the second surface has no submodel defined, it will be using the background values from the tomography.
smoothing: Smoothing boundary that define where velocity models should be smoothly transitioned between basins and background model (ie. tomography)
For the completeness, the below are the definition of CheviotBasement

```
surface:
  - name: CheviotBasement
    path: SI_BASINS/Cheviot_Basement_WGS84_v0p0.in
```

and canterbury1d_v2

```
submodel:
    - name: canterbury1d_v2
      type: vm1d
```
as it is vm1d type, it has a separate definition in the registry for the path to the data.

```
vm1d:
  - name: canterbury1d_v2
    path: 1D_Velocity_Model/Cant1D_v2.fd_modfile
```

Why canterbury1d_v2 model for most basins not in Canterbury region? It is the most widely used 1D velocity model and considered a good representation for other parts of New Zealand.



## References

Ethan M. Thomson, Brendon A. Bradley & Robin L. Lee (2020) Methodology and computational implementation of a New Zealand Velocity Model (NZVM2.0) for broadband ground motion simulation, New Zealand Journal of Geology and Geophysics, 63:1, 110-127, DOI: 10.1080/00288306.2019.1636830

Donna Eberhart-Phillips, Martin Reyners, Stephen Bannister, Mark Chadwick, Susan Ellis; Establishing a Versatile 3-D Seismic Velocity Model for New Zealand. Seismological Research Letters 2010; 81 (6): 992–1000. doi: https://doi.org/10.1785/gssrl.81.6.992

Donna Eberhart-Phillips, Stephen Bannister, Martin Reyners, and Stuart Henrys. “New Zealand Wide Model 2.2 Seismic Velocity and Qs and Qp Models for New Zealand”. Zenodo, May 1, 2020. https://doi.org/10.5281/zenodo.3779523.


