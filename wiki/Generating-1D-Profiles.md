# Generating 1D Velocity Profiles

## Overview

The `generate_1d_profiles.py` script generates multiple 1D velocity profiles based on user-supplied coordinates and depth parameters. It reads profile locations from a CSV file, loads the required velocity model datasets, and outputs velocity/density profiles and surface depths for each location.

---

## Basic Usage

The script requires one main input and accepts several optional parameters:

1. **Location CSV** (required): A CSV file listing the profile locations and depth parameters. Each row should contain: `id, lon, lat, zmin, zmax, spacing`.
2. **Model Version**: The velocity model version to use (default: `2.09`).
3. **Output Directory**: Where the generated profile files will be saved (default: same directory as the location CSV).

### Example Command

```bash
python velocity_modelling/scripts/generate_1d_profiles.py locations.csv \
  --model-version 2.08 \
  --out-dir /path/to/output/ \
  --topo-type SQUASHED_TAPERED \
  --min-vs 0.5
```

### Minimal Usage

```bash
# Uses defaults: model version 2.09, outputs to same directory as CSV
python velocity_modelling/scripts/generate_1d_profiles.py locations.csv
```

---

### Required Arguments

- **location_csv**: CSV file with profile parameters (positional argument, required)

### Optional Arguments

- **--out-dir**: Directory for output files (default: parent directory of location CSV)
- **--model-version**: Model version to use (default: 2.09)
- **--min-vs**: Minimum shear wave velocity (default: 0.0)
- **--topo-type**: Topography type (`TRUE`, `BULLDOZED`, `SQUASHED`, `SQUASHED_TAPERED`; default: `TRUE`)
- **--custom-depth**: Text file with custom depth points (overrides zmin, zmax, spacing in CSV)
- **--nzcvm-registry**: Path to the model registry file (default: nzcvm_registry.yaml)
- **--nzcvm-data-root**: Override the default data root directory
- **--log-level**: Logging level (default: INFO)

---

## Input File Format

### Location CSV Example

```csv
id,lon,lat,zmin,zmax,spacing
809,172.49674504501075,-43.57101252665522,-0.013,2.573,0.03923
158,172.7244163,-43.51491883,0,3,0.05
...
```

- **id**: Unique identifier for the profile (used in output filenames)
- **lon, lat**: Longitude and latitude of the profile location
- **zmin, zmax**: Minimum and maximum depth (in km, negative values for above sea level)
- **spacing**: Depth interval (in km)

### Custom Depth File (Optional)

A plain text file with one depth value per line (in km). If provided, these depths override the zmin, zmax, and spacing from the CSV for all profiles.
(Example: `custom_depths.txt`)
```angular2html
        0.0
        0.1
        0.5
        1.0
        5.0
```

---

## Output Files

For each profile location, two files are generated in the output directory:

1. **Profile_(id).txt**: The velocity profile at the specified location.
2. **ProfileSurfaceDepths_(id).txt**: The elevations of global and basin surfaces at the location.

By default, these files are created in the same directory as the input CSV file. Use `--out-dir` to specify a different location.

### Example: Profile_158.txt

```
Properties at Lat : -43.514919 Lon: 172.724416 (On Mesh Lat: -43.514913 Lon: 172.724411)
Model Version: 2.08
Topo Type: SQUASHED_TAPERED
Minimum Vs: 0.500000
Elevation (km) Vp (km/s)   Vs (km/s)   Rho (g/cm^3)
-0.000000      1.800000    0.500000    1.810000
-0.050000      1.800000    0.500000    1.810000
...
-3.000000 4.813934 2.762850 2.578671
```

### Example: ProfileSurfaceDepths_158.txt

```
Surface Elevation (in m) at Lat : -43.514919 Lon: 172.724416 (On Mesh Lat: -43.514913 Lon: 172.724411)

Global Surfaces
Surface_name Elevation (m)
- posInf     1000000.000000
- NZ_DEM_HD  1.807371
- negInf    -1000000.000000

Basin surfaces (if applicable)

Canterbury_v19p1
- CantDEM    1.882188
- Canterbury_Pliocene_46_WGS84_v8p9p18    -202.709242
...
```

---

## Notes

- The script supports all topography types used in the NZCVM: `TRUE`, `BULLDOZED`, `SQUASHED`, `SQUASHED_TAPERED`.
- The model version string (e.g., `2.08`) must correspond to a YAML file in the model_version directory (e.g., `2p08.yaml`).
- The output files are named after the profile `id` in the CSV.
- By default, output files are saved in the same directory as the input CSV file.

---

## Troubleshooting

- Ensure the location CSV file exists and has the correct format.
- Check that the model version YAML exists and is correctly named.
- If specifying a custom output directory, ensure you have write permissions.

---
