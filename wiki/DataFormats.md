# Data Formats in NZCVM

This page provides detailed information about the various data formats used in the NZCVM for surfaces, boundaries, tomography, 1D velocity models, and smoothing data.

## Surface Data Format

Two data formats are used for surface data: ASCII grid files and HDF5 grid files, which contain elevation or depth data on a 2D grid.

ASCII grid files have a `.in` extension and are compatible with old C code. HDF5 grid files have a `.h5` extension and are the preferred format for new data.
All the surface data files are stored in the `cvm/data/global/surface` and `cvm/data/regional/<basin_name>/` directories, and we have both formats available.


### ASCII Grid Format
Surface data files (`.in` extension) contain elevation or depth data on a 2D grid.

#### Format Specification

```
ny (number of latitudes) nx (number of longitues)
lat_1 lat_2 lat_3.....lat_ny
lon_1 lon_2 lon_3.....lon_nx
z_value_1_1 z_value_1_2 ... z_value_1_nx
z_value_2_1 z_value_2_2 ... z_value_2_nx
...
z_value_ny_1 z_value_ny_2 ... z_value_ny_nx
```

Where:
- `nx` and `ny` are the number of grid points in the x and y directions
- `z_value_lat_lon` is the elevation or depth value at grid point (lat, lon)

If the data file has a missing value, it will warn the user and pad with zeros to match the required data length during the run time.

```
2025-03-25 21:13:21,015 - nzcvm - WARNING - In /Users/sungbae/velocity_modelling/velocity_modelling/cvm/data/regional/Canterbury/Canterbury_Miocene_WGS84.in: Data length mismatch - got 150800, expected 150801. Missing data will be padded with 0.
```
Note: This is likely due to clerical error during data preparation. We chose to pad with zeros to match the behaviour original C code, but this may lead to a undesirable outcome.

#### Example

```
180 227 
-45.590000 -45.585000 ... -44.695000
169.270000 169.275000 ... 170.400000
14.313100 14.595800 ...
...
15.695500 16.386100 ...
```

### HDF5 Grid Format

Surface data files with the `.h5` extension are stored in HDF5 format, offering an efficient structure for large datasets with built-in compression.

#### Format Specification

The HDF5 surface files include the following components:

- **Attributes**:
  - `nrows`: Number of latitude points (integer)
  - `ncols`: Number of longitude points (integer)

- **Datasets**:
  - `latitude`: 1D array of latitude values [shape: (nrows,)]
  - `longitude`: 1D array of longitude values [shape: (ncols,)]
  - `elevation`: 2D array of elevation or depth values [shape: (nrows, ncols)]

All datasets are typically stored with gzip compression to reduce file size while preserving data integrity. The provided tool `tools/surface_ascii2h5.py` can be used to convert ASCII format files to HDF5 format.

#### Access Example

```python
import h5py

with h5py.File('surface_data.h5', 'r') as f:
    # Access attributes
    nrows = f.attrs['nrows']
    ncols = f.attrs['ncols']
    
    # Access datasets
    latitude = f['latitude'][:]
    longitude = f['longitude'][:]
    elevation = f['elevation'][:]
    
    print(f"Number of rows: {nrows}, Number of columns: {ncols}")
    print(f"Latitude range: {latitude[0]} to {latitude[-1]}")
    print(f"Longitude range: {longitude[0]} to {longitude[-1]}")
```

## Boundary Data Format

Boundary files (typically `.txt` extension) define the geographical boundaries of basins and other geological features as closed polygons.

### Format Specification

```
lon_1 lat_1
lon_2 lat_2
...
lon_n lat_n
```

Where:
- `lon_i` and `lat_i` are the longitude and latitude coordinates of point i
The first location should be the same as the last to form a closed polygon. ie. (lon_1, lat_1)=(lon_n, lat_n)
If the boundary data is found to be not closed, it will throw an error and program will terminate.

### Example

```
172.5000 -43.5000
172.6000 -43.5000
172.7000 -43.4500
172.7000 -43.3500
172.6000 -43.3000
172.5000 -43.3000
172.4000 -43.3500
172.4000 -43.4500
...
172.5000 -43.5000
```

## Tomography Data Format

Tomography data is stored in HDF5 format (`.h5` extension). These files contain 3D grids of velocity values derived from seismic tomography.

### Structure Overview
```
/
├── "elevation1" (e.g., "-750" or "0.25")
│   ├── latitudes [array of float values]
│   ├── longitudes [array of float values]
│   ├── vp [2D array - shape(nlat, nlon)]
│   ├── vs [2D array - shape(nlat, nlon)]
│   └── rho [2D array - shape(nlat, nlon)]
├── "elevation2" (another elevation layer)
│   ├── latitudes [array of float values]
│   ├── longitudes [array of float values]
│   ├── vp [2D array - shape(nlat, nlon)]
│   ├── vs [2D array - shape(nlat, nlon)]
│   └── rho [2D array - shape(nlat, nlon)]
└── ... (additional elevation groups)
```
### Structure Details
- Root Level: Contains groups named after elevation values, (e.g., "-750" or "0.25") 
- Elevation Groups: Each elevation group contains:

  - latitudes: 1D array of latitude coordinates
  - longitudes: 1D array of longitude coordinates
  - vp: 2D array of P-wave velocities at grid points [nlat × nlon]
  - vs: 2D array of S-wave velocities at grid points [nlat × nlon]
  - rho: 2D array of density values at grid points [nlat × nlon]
- Grid Structure: The velocity and density data are arranged in 2D grids where:
    - First dimension corresponds to the latitude points
    - Second dimension corresponds to the longitude points
    - Values represent the property (vp, vs, or rho) at that lat/lon coordinate


### Access Example

```python
import h5py

with h5py.File('2010_NZ.h5', 'r') as f:
    # List available elevations
    elevations = list(f.keys())
    print(f"Available elevations: {elevations}")
    
    # Access data for a specific elevation
    elev = elevations[0]  # For example, get the first elevation
    
    # Get coordinate arrays
    latitudes = f[elev]['latitudes'][:]
    longitudes = f[elev]['longitudes'][:]
    
    # Get velocity and density data
    vp = f[elev]['vp'][:]
    vs = f[elev]['vs'][:]
    rho = f[elev]['rho'][:]
    
    # Example: Access value at specific lat/lon index
    i_lat, i_lon = 10, 20
    vs_value = vs[i_lat, i_lon]
    print(f"S-wave velocity at lat={latitudes[i_lat]}, lon={longitudes[i_lon]}: {vs_value} km/s")
```

## 1D Velocity Model Format

1D velocity models (`.fd_modfile` extension) define velocity profiles that vary only with depth. These files are used for simpler basin models.

### Format Specification

```
header
vp_1 vs_1 rho_1 qp_1 qs_1 depth_1
vp_2 vs_2 rho_2 qp_2 qs_2 depth_2 
...
vp_n vs_n rho_n qp_n qs_n depth_n 
```

Where:
- `vp_i` is the P-wave velocity in km/s
- `vs_i` is the S-wave velocity in km/s
- `rho_i` is the density in g/cm^3
- `qp_i` is the P-wave quality factor
- `qs_i` is the S-wave quality factor
- `depth_i` is the depth in kilometers
### Example

```
DEF HST
  1.80   0.50   1.81   50.0   25.0    0.400
  1.90   0.58   1.86   58.0   29.0    0.600
  2.03   0.66   1.92   66.0   33.0    0.800
  2.14   0.74   1.97   74.0   37.0    1.000
  2.20   0.80   1.99   80.0   40.0    1.200
  2.40   1.01   2.06  101.0   50.5    1.400
  2.70   1.22   2.15  122.0   61.0    1.600
...
```

## Smoothing Data Format

Smoothing files define regions where velocity models should be smoothly transitioned between basins and the background model.

### Format Specification

```
lon_1 lat_1
lon_2 lat_2
...
lon_n lat_n
```

Where:
- `lon_i` and `lat_i` are the longitude and latitude coordinates of point i

This is the same format as the boundary, but doesn't have a requirement of being the first and last point to be identical.

### Example

```
170.00831948 -46.23795676
170.00960087 -46.23802830
170.01082825 -46.23830416
170.01198789 -46.23870333
170.01310530 -46.23915898
170.01420524 -46.23963506
170.01529993 -46.24011699
...
```

## Data Locations

The data files used by the NZCVM are located in the following directories:

- **Surface data**: `cvm/data/global/surface` and `cvm/data/regional/<basin_name>/`
- **Boundary data**: `cvm/data/regional/<basin_name>/`
- **Tomography data**: `cvm/data/global/tomography/`
- **1D velocity models**: `cvm/data/global/vm1d/`
- **Smoothing data**: `cvm/data/regional/<basin_name>/`
