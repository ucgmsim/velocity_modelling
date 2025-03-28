# Tomography Models in NZCVM

The tomography models in the NZCVM provide the background velocity structure for New Zealand. These models are derived from seismic travel-time data and offer a lower-resolution (~10km) representation of the subsurface velocity structure.

## Available Tomography Models

The NZCVM currently supports the following tomography models:

1. **2010_NZ**: Based on Eberhart-Phillips et al. (2010)
2. **2010_NZ_OFFSHORE**: Extension of the 2010 model with a basin submodel applied to the offshore region
3. **2020_NZ**: Based on Eberhart-Phillips et al. (2020)
4. **2020_NZ_OFFSHORE**: Extension of the 2020 model with a basin submodel applied to the offshore region
5. **2020_NZ_OFFSHORE_NO_BASIN** : Extension of the 2020 model with offshore region but no basin submodel applied

## Tomography Model Definition

Tomography models are defined in the `nzcvm_registry.yaml` file. Here's an example of a tomography model definition:

```yaml
tomography:
  - name: 2010_NZ_OFFSHORE
    nElev: 20
    elev: [ 15, 1, -3, -8, -15, -23, -30, -38, -48, -65, -85, -105, -130, -155, -185, -225, -275, -370, -620, -750 ]
    vs30_path: global/vs30/NZ_Vs30_HD_With_Offshore.h5
    special_offshore_tapering: true
    path: global/tomography/2010_NZ.h5
```

The key components of a tomography model definition are:

- **name**: Identifier for the tomography model
- **nElev**: Number of elevation levels in the model
- **elev**: Array of elevation values (in kilometers) for the model
- **vs30_path**: Path to the Vs30 (shear wave velocity in the top 30 meters) data file
- **special_offshore_tapering**: Whether to apply basin submodel in offshore regions (default submodel: canterbury1d_submod)
- **path**: Path to the tomography model data file (in HDF5 format)

## Usage in Model Versions

Tomography models are specified in the model version configuration files. For example, in `2p03.yaml`:

```yaml
tomography: 2010_NZ_OFFSHORE
```

This indicates that the 2p03 model version uses the 2010_NZ_OFFSHORE tomography model.

## Tomography Submodels

Tomography submodels are used to compute velocity values at specific locations based on the tomography model data. These submodels are defined in the `nzcvm_registry.yaml` file and are associated with surfaces in the model version configuration.

```yaml
submodel:
  - name: ep_tomography_submod_v2010
    type: tomography
    module: ep_tomography_submod_v2010
```

This submodel is a type of `tomography`, and it retrieves more information associated with the `tomography` value from the registry (in this case, `2010_NZ_OFFSHORE`). The `module` is the name of the accompanied Python code that prescribes how to calculate velocity at the location within the region below the `surface`.

## Data Format

Tomography model data is stored in HDF5 format. For details on the structure and contents of these files, see the [Data Formats](DataFormats.md) page.

## References

- Donna Eberhart-Phillips, Martin Reyners, Stephen Bannister, Mark Chadwick, Susan Ellis; Establishing a Versatile 3-D Seismic Velocity Model for New Zealand. Seismological Research Letters 2010; 81 (6): 992–1000. doi: https://doi.org/10.1785/gssrl.81.6.992

- Donna Eberhart-Phillips, Stephen Bannister, Martin Reyners, and Stuart Henrys. "New Zealand Wide Model 2.2 Seismic Velocity and Qs and Qp Models for New Zealand". Zenodo, May 1, 2020. https://doi.org/10.5281/zenodo.3779523.
