# Extracting Cross-Sections

This page describes how to use the `extract_cross_section.py` script to extract cross-sections from a velocity model in HDF5 format.

## Overview

The `extract_cross_section.py` script allows you to extract a vertical or horizontal cross-section from a 3D velocity model. The cross-section can be specified either by latitude/longitude coordinates or by grid indices. The output can be visualized as an image or exported as a CSV file.

## Usage

```sh
python extract_cross_section.py <h5file> [options]
```

### Required Arguments

- `<h5file>`: Path to the HDF5 velocity model file.

### Options

- `--lat1 <lat1> --lon1 <lon1> --lat2 <lat2> --lon2 <lon2>`  
  Specify the start and end points of the cross-section using geographic coordinates.

- `--x1 <x1> --y1 <y1> --x2 <x2> --y2 <y2>`  
  Specify the start and end points using grid indices.

- `--property <property_name>`  
  The property to extract (e.g., `vs`, `vp`, `rho`).

- `--xaxis <xaxis>`  
  The axis for the cross-section (`auto`, `lat`, `lon`).

- `--n_points <n_points>`  
  Number of points along the cross-section (default is 350).

- `--vmin <vmin> --vmax <vmax>`  
  Minimum and maximum values for color scaling.

- `--max_depth <max_depth>`  
  Maximum depth to extract (default: None, extracts full depth).

- `--cmap <cmap>`  
  Colormap for visualization (default: `jet`).

- `--png <output_png>`  
  Prefix of the output PNG image file (e.g., `AA` for `AA_map.png and AA_plot.png`).

- `--csv <csv_output>`  
  Output CSV file.

## Example

Extract a cross-section between two geographic points and save as PNG and CSV:

```sh
python extract_cross_section.py --lat1 -43.5 --lon1 171 --lat2 -44.5 --lon2 172 --vmax 9  --xaxis lat \ 
--png AA  velocity_model.h5
```

## Output

- **PNG Image**: Visual representation of the cross-section. If `--png` is specified, the value is used as a prefix for the output files:
  - `{prefix}_map.png`: Map view of the cross-section.
  - `{prefix}_plot.png`: Plot view of the cross-section.
<p align="center">
  <img src="https://github.com/user-attachments/assets/f5d953f5-e2b6-482d-9b32-7d3b3953b4fb" alt="AA_map" width="50%">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/12c0693a-1581-4e3e-ad2c-fb468f184768" alt="AA_plot" width="50%">
</p>

- **CSV File**: Tabular data of the extracted cross-section.


## Notes

- The script supports both geographic and grid-based specification of cross-section endpoints.
- For more details on available properties and options, run:
  ```sh
  python extract_cross_section.py --help
  ```

## See Also

- [Generating 3D Model](Generating-3D-Model.md)
- [Generating 1D Profiles](Generating-1D-Profiles.md)
- [Output Formats](OutputFormats.md)
