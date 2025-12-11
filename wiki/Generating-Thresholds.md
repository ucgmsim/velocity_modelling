# Generating Thresholds

## Overview

The `generate_thresholds.py` script computes velocity and depth threshold values (Vs30, Vs500, Z1.0, Z2.5) for specified station locations. It reads station coordinates from a CSV file, queries the NZCVM velocity model, and outputs the requested threshold parameters. When Z-thresholds (Z1.0, Z2.5) are requested, the script uses precomputed basin membership to assign appropriate sigma values.

---

## Basic Usage

The script requires one main input:

1. **Locations CSV**: A CSV file listing station locations. Default format: `id,longitude,latitude` (one station per row).

The script will compute the requested threshold values using the specified NZCVM model version and save results to a CSV file.

### Example Command

```bash
python velocity_modelling/scripts/generate_thresholds.py \
  locations.csv \
  --model-version 2.07 \
  --threshold-type Z1.0 \
  --threshold-type Z2.5 \
  --topo-type SQUASHED_TAPERED \
  --out-dir /path/to/output/
```

### Legacy Station File Support

For backward compatibility with legacy station files (format: `lon lat name`), use:

```bash
python velocity_modelling/scripts/generate_thresholds.py \
  stations.ll \
  --lon-index 0 --lat-index 1 --name-index 2 --sep " " \
  --model-version 2.07
```

---

## Required Arguments

- **locations_csv**: Path to the CSV file containing station locations (positional argument)

---

## Optional Arguments

- **--model-version**: NZCVM model version to use (default: `2.07`)
- **--threshold-type**: Threshold types to compute. Can be specified multiple times. Options:
  - `VS30`: Time-averaged shear-wave velocity to 30m depth
  - `VS500`: Time-averaged shear-wave velocity to 500m depth
  - `Z1.0`: Depth to 1.0 km/s shear-wave velocity horizon
  - `Z2.5`: Depth to 2.5 km/s shear-wave velocity horizon
  - If not specified, defaults to `[Z1.0, Z2.5]`
- **--topo-type**: Topography handling method (default: `SQUASHED`). Options:
  - `TRUE`: Use actual topography from DEM
  - `BULLDOZED`: Flatten topography to sea level
  - `SQUASHED`: Compress topography while preserving relative elevations
  - `SQUASHED_TAPERED`: Squashed topography with edge tapering
- **--name-index**: Column index for station names (0-based, default: 0)
- **--lat-index**: Column index for latitude (0-based, default: 1)
- **--lon-index**: Column index for longitude (0-based, default: 2)
- **--sep**: Column separator/delimiter (default: ",")
- **--skip-rows**: Number of header rows to skip at the beginning of the file (default: 0)
- **--out-dir**: Output directory for results (default: current working directory)
- **--write-header / --no-write-header**: Write CSV output with header row (default: enabled). Use `--no-write-header` to disable.
- **--nzcvm-registry**: Optional path to custom nzcvm_registry.yaml file (default: registry under data root)
- **--nzcvm-data-root**: Override for data root directory (default: use configured default)
- **--log-level**: Logging level, e.g., "INFO", "DEBUG" (default: INFO)

---

## Input File Formats

### Default CSV Format (Recommended)

```csv
id,lon,lat
STATION_A,172.7244163,-43.51491883
STATION_B,174.7765285,-41.28908245
STATION_C,168.6626837,-44.99783415
```

Default column order: `[id, longitude, latitude]`

### Legacy Station File Format

```
172.7244163 -43.51491883 STATION_A
174.7765285 -41.28908245 STATION_B
168.6626837 -44.99783415 STATION_C
```

Use with: `--lon-index 0 --lat-index 1 --name-index 2 --sep " "`

### Custom Formats

The script supports any delimited format by specifying column indices and separator:

**Tab-separated with custom order:**
```
STATION_A	-43.515	172.724
STATION_B	-41.289	174.777
```
Use with: `--name-index 0 --lat-index 1 --lon-index 2 --sep "\t"`

**Pipe-separated:**
```
-43.515|172.724|STATION_A
-41.289|174.777|STATION_B
```
Use with: `--lat-index 0 --lon-index 1 --name-index 2 --sep "|"`

**CSV with header row:**
```csv
name,latitude,longitude
STATION_A,-43.515,172.724
STATION_B,-41.289,174.777
```
Use with: `--name-index 0 --lat-index 1 --lon-index 2 --skip-rows 1`

---

## Output Files

The script generates a CSV file in the output directory containing the computed threshold values for each station.

### Output Filename

The output file uses the input filename with "_thresholds" suffix and ".csv" extension to avoid overwriting the input file.

**Examples**: 
- Input: `locations.csv` → Output: `locations_thresholds.csv`
- Input: `stations.txt` → Output: `stations_thresholds.csv`
- Input: `my_sites.dat` → Output: `my_sites_thresholds.csv`

### Example Output

```csv
Station_Name,Vs30(m/s),Vs500(m/s),Z1.0(km),Z2.5(km),sigma
HNPS,250.5,450.2,0.195,5.625,0.300
TRMS,180.3,380.1,0.055,0.625,0.500
NSPS,220.8,420.5,0.105,6.625,0.300
MWZ,300.2,520.4,0.045,0.275,0.500
```

The CSV includes:
- **Station_Name**: Station identifier from input file
- **Threshold columns**: One column for each requested threshold type, formatted as `<type>_<threshold>(<units>)`
  - For Vs-thresholds: `Vs30(m/s)`, `Vs500(m/s)` 
  - For Z-thresholds: `Z1.0(km)`, `Z2.5(km)`
- **sigma**: Uncertainty parameter based on basin membership (only included when Z-thresholds are requested)
  - `0.300`: Station is inside a sedimentary basin
  - `0.500`: Station is outside any basin

---

## Threshold Definitions

- **Vs30**: Time-averaged shear-wave velocity over the top 30 meters. Used as a proxy for site amplification effects.
- **Vs500**: Time-averaged shear-wave velocity over the top 500 meters. Provides broader characterization of site conditions.
- **Z1.0**: Depth (in km) to the 1.0 km/s shear-wave velocity isosurface. Important for basin amplification modeling.
- **Z2.5**: Depth (in km) to the 2.5 km/s shear-wave velocity isosurface. Characterizes sedimentary basin depth.

### Topography Handling

The `--topo-type` parameter controls how surface topography is handled in the velocity model:

- **TRUE**: Uses actual topography from the Digital Elevation Model (DEM). This preserves the real surface geometry but may result in irregular model grids.
- **BULLDOZED**: Flattens all topography to sea level (0m elevation). This creates a uniform flat surface, simplifying the model but removing topographic effects.
- **SQUASHED**: Compresses the topography while preserving relative elevation differences. This maintains topographic variation while creating a more regular grid.
- **SQUASHED_TAPERED**: Similar to SQUASHED but with edge tapering to reduce boundary effects. This is often preferred for regional models.

The choice of topography type can affect threshold calculations, particularly for stations at high elevations or in areas with significant topographic relief.

### Sigma Values (Basin Membership)

When Z-thresholds (Z1.0, Z2.5) are computed, the script automatically determines basin membership for each station and assigns a sigma value:

- **sigma = 0.300**: Station is located inside a sedimentary basin. Lower uncertainty due to better-constrained basin geometry.
- **sigma = 0.500**: Station is located outside any basin. Higher uncertainty due to less constrained subsurface structure.

These sigma values represent the uncertainty in the depth predictions and are used in ground motion simulation workflows for uncertainty quantification.

---

## Examples

### Example 1: Default CSV Format
```bash
# Create locations file
echo "id,lon,lat" > locations.csv
echo "STATION_A,172.724,-43.515" >> locations.csv
echo "STATION_B,174.777,-41.289" >> locations.csv

# Generate thresholds
generate_thresholds locations.csv --model-version 2.07 --threshold-type Z1.0 --threshold-type Z2.5
```

### Example 2: Legacy Station File Format
```bash
# Create legacy station file
echo "172.724 -43.515 STATION_A" > stations.ll
echo "174.777 -41.289 STATION_B" >> stations.ll

# Generate thresholds with format specification
generate_thresholds stations.ll \
  --lon-index 0 --lat-index 1 --name-index 2 --sep " " \
  --model-version 2.07
```

### Example 3: Custom Tab-Separated Format
```bash
# Create tab-separated file
printf "STATION_A\t-43.515\t172.724\n" > data.tsv
printf "STATION_B\t-41.289\t174.777\n" >> data.tsv

# Generate thresholds
generate_thresholds data.tsv \
  --name-index 0 --lat-index 1 --lon-index 2 --sep "\t" \
  --threshold-type VS30 --threshold-type Z1.0
```

---

## Notes

- The script requires the NZCVM model files to be properly installed and configured.
- Multiple `--threshold-type` options can be specified to compute multiple thresholds in a single run.
- If no `--threshold-type` is specified, the script defaults to computing Z1.0 and Z2.5.
- The default topography type is `SQUASHED`, which provides a good balance between preserving topographic effects and maintaining model stability.
- The model version string (e.g., `2.07`) must correspond to a valid NZCVM model version. See [Model Versions](Model-Versions.md) for details
- Station coordinates should be in WGS84 decimal degrees.
- The **sigma column is only included in the output when Z-thresholds (Z1.0, Z2.5) are requested**. Vs-threshold computations (Vs30, Vs500) do not include sigma values.
- The script handles basin membership automatically when computing Z-thresholds.
- The output CSV filename is derived from the input filename with "_thresholds" suffix to prevent overwriting the input file.
- Column indices are 0-based (first column = 0, second column = 1, etc.).

---

## Troubleshooting

- **File format errors**: Ensure the input file matches the specified column indices and separator.
- **Invalid column indices**: Check that the specified indices exist in your input file (use 0-based indexing).
- **Model version not found**: Check that the specified model version exists in your NZCVM installation.
- **Invalid topography type**: Ensure the `--topo-type` parameter uses one of the valid options: TRUE, BULLDOZED, SQUASHED, SQUASHED_TAPERED.
- **Invalid coordinates**: Ensure latitude values are between -90 and 90, and longitude values are valid decimal degrees.
- **Missing data**: If certain stations return null values, they may be outside the model domain or in regions with insufficient data coverage.
- **Registry errors**: If using a custom registry, ensure the path to `nzcvm_registry.yaml` is correct.
- **Output file conflicts**: If a CSV file with the same name already exists in the output directory, it will be overwritten.

---
