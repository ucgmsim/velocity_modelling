# Generating Thresholds

## Overview

The `generate_thresholds.py` script computes velocity and depth threshold values (Vs30, Vs500, Z1.0, Z2.5) for specified station locations. It reads station coordinates from a file, queries the NZCVM velocity model, and outputs the requested threshold parameters. When Z-thresholds (Z1.0, Z2.5) are requested, the script uses precomputed basin membership to assign appropriate sigma values.

---

## Basic Usage

The script requires one main input:

1. **Station File**: A text file listing station locations with format: `lon lat station_name` (one station per line).

The script will compute the requested threshold values using the specified NZCVM model version and save results to a CSV file.

### Example Command

```bash
python velocity_modelling/scripts/generate_thresholds.py \
  station_file \
  --model-version 2.07 \
  --vs-type Z1.0 \
  --vs-type Z2.5 \
  --out-dir /path/to/output/
```

---

## Required Arguments

- **station_file**: Path to the station file containing station locations (positional argument)

---

## Optional Arguments

- **--model-version**: NZCVM model version to use (default: `2.07`)
- **--vs-type**: Threshold types to compute. Can be specified multiple times. Options:
  - `VS30`: Time-averaged shear-wave velocity to 30m depth
  - `VS500`: Time-averaged shear-wave velocity to 500m depth
  - `Z1.0`: Depth to 1.0 km/s shear-wave velocity horizon
  - `Z2.5`: Depth to 2.5 km/s shear-wave velocity horizon
  - If not specified, defaults to `[Z1.0, Z2.5]`
- **--out-dir**: Output directory for results (default: current working directory)
- **--no-header**: If specified, write CSV output without a header row
- **--nzcvm-registry**: Optional path to custom nzcvm_registry.yaml file (default: registry under data root)
- **--nzcvm-data-root**: Override for data root directory (default: use configured default)
- **--log-level**: Logging level, e.g., "INFO", "DEBUG" (default: INFO)

---

## Input File Format

### Station File Example

```
172.7244163 -43.51491883 STATION_A
174.7765285 -41.28908245 STATION_B
168.6626837 -44.99783415 STATION_C
```

Each line contains three space-separated values:
- **longitude**: Station longitude (decimal degrees)
- **latitude**: Station latitude (decimal degrees)
- **station_name**: Unique identifier for the station

---

## Output Files

The script generates a CSV file in the output directory containing the computed threshold values for each station.

### Output Filename

The output file uses the same name as the input station file, with the extension changed to `.csv`.

**Examples**: 
- Input: `stations.txt` → Output: `stations.csv`
- Input: `my_station_list.dat` → Output: `my_station_list.csv`
- Input: `site_locations.txt` → Output: `site_locations.csv`

### Example Output

```csv
Station_Name,Vs_30(m/s),Vs_500(m/s),Z_Z1.0(km),Z_Z2.5(km),sigma
HNPS,250.5,450.2,0.195,5.625,0.300
TRMS,180.3,380.1,0.055,0.625,0.500
NSPS,220.8,420.5,0.105,6.625,0.300
MWZ,300.2,520.4,0.045,0.275,0.500
```

The CSV includes:
- **Station_Name**: Station identifier from input file
- **Threshold columns**: One column for each requested threshold type, formatted as `<type>_<threshold>(<units>)`
  - For Vs-thresholds: `Vs_30(m/s)`, `Vs_500(m/s)` 
  - For Z-thresholds: `Z_Z1.0(km)`, `Z_Z2.5(km)`
- **sigma**: Uncertainty parameter based on basin membership (only included when Z-thresholds are requested)
  - `0.300`: Station is inside a sedimentary basin
  - `0.500`: Station is outside any basin

---

## Threshold Definitions

- **Vs30**: Time-averaged shear-wave velocity over the top 30 meters. Used as a proxy for site amplification effects.
- **Vs500**: Time-averaged shear-wave velocity over the top 500 meters. Provides broader characterization of site conditions.
- **Z1.0**: Depth (in km) to the 1.0 km/s shear-wave velocity isosurface. Important for basin amplification modeling.
- **Z2.5**: Depth (in km) to the 2.5 km/s shear-wave velocity isosurface. Characterizes sedimentary basin depth.

### Sigma Values (Basin Membership)

When Z-thresholds (Z1.0, Z2.5) are computed, the script automatically determines basin membership for each station and assigns a sigma value:

- **sigma = 0.300**: Station is located inside a sedimentary basin. Lower uncertainty due to better-constrained basin geometry.
- **sigma = 0.500**: Station is located outside any basin. Higher uncertainty due to less constrained subsurface structure.

These sigma values represent the uncertainty in the depth predictions and are used in ground motion simulation workflows for uncertainty quantification.

---

## Notes

- The script requires the NZCVM model files to be properly installed and configured.
- Multiple `--vs-type` options can be specified to compute multiple thresholds in a single run.
- If no `--vs-type` is specified, the script defaults to computing Z1.0 and Z2.5.
- The model version string (e.g., `2.07`) must correspond to a valid NZCVM model version.
- Station coordinates should be in WGS84 decimal degrees.
- The **sigma column is only included in the output when Z-thresholds (Z1.0, Z2.5) are requested**. Vs-threshold computations (Vs30, Vs500) do not include sigma values.
- The script handles basin membership automatically when computing Z-thresholds.
- The output CSV filename is derived from the input station filename (same name, .csv extension).

---

## Troubleshooting

- **File format errors**: Ensure the station file has exactly three space-separated columns per line.
- **Model version not found**: Check that the specified model version exists in your NZCVM installation.
- **Missing data**: If certain stations return null values, they may be outside the model domain or in regions with insufficient data coverage.
- **Registry errors**: If using a custom registry, ensure the path to `nzcvm_registry.yaml` is correct.
- **Output file conflicts**: If a CSV file with the same name already exists in the output directory, it will be overwritten.

---
