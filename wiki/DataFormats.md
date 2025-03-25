# Data Formats in NZCVM

This page provides detailed information about the various data formats used in the NZCVM for surfaces, boundaries, tomography, 1D velocity models, and smoothing data.

## Surface Data Format

Surface data files (`.in` extension) contain elevation or depth data on a 2D grid. These files are used to define the top and bottom surfaces of basins and other geological features.

### Format Specification

```
nx ny
lon_min lon_max lat_min lat_max
z_value_1_1 z_value_1_2 ... z_value_1_ny
z_value_2_1 z_value_2_2 ... z_value_2_ny
...
z_value_nx_1 z_value_nx_2 ... z_value_nx_ny
```

Where:
- `nx` and `ny` are the number of grid points in the x and y directions
- `lon_min`, `lon_max`, `lat_min`, and `lat_max` define the geographic extent of the grid
- `z_value_i_j` is the elevation or depth value at grid point (i, j)

### Example

```
100 120
170.0 172.0 -44.0 -42.0
125.6 125.3 ... 124.8
125.4 125.2 ... 124.7
...
123.8 123.7 ... 123.0
```

## Boundary Data Format

Boundary files (typically `.txt` extension) define the geographical boundaries of basins and other geological features as closed polygons.

### Format Specification

```
num_points
lon_1 lat_1
lon_2 lat_2
...
lon_n lat_n
```

Where:
- `num_points` is the number of points in the boundary
- `lon_i` and `lat_i` are the longitude and latitude coordinates of point i

### Example

```
8
172.5000 -43.5000
172.6000 -43.5000
172.7000 -43.4500
172.7000 -43.3500
172.6000 -43.3000
172.5000 -43.3000
172.4000 -43.3500
172.4000 -43.4500
```

## Tomography Data Format

Tomography data is stored in HDF5 format (`.h5` extension). These files contain 3D grids of velocity values derived from seismic tomography.

### Format Specification

The HDF5 file contains the following datasets:
- `lat`: 1D array of latitude values
- `lon`: 1D array of longitude values
- `depth`: 1D array of depth values
- `vp`: 3D array of P-wave velocity values
- `vs`: 3D array of S-wave velocity values
- `rho`: 3D array of density values

### Access Example

```python
import h5py

with h5py.File('tomography_file.h5', 'r') as f:
    lat = f['lat'][:]
    lon = f['lon'][:]
    depth = f['depth'][:]
    vp = f['vp'][:]
    vs = f['vs'][:]
    rho = f['rho'][:]
```

## 1D Velocity Model Format

1D velocity models (`.fd_modfile` extension) define velocity profiles that vary only with depth. These files are used for simpler basin models.

### Format Specification

```
depth_1 vp_1 vs_1 rho_1 qp_1 qs_1
depth_2 vp_2 vs_2 rho_2 qp_2 qs_2
...
depth_n vp_n vs_n rho_n qp_n qs_n
```

Where:
- `depth_i` is the depth in kilometers
- `vp_i` is the P-wave velocity in km/s
- `vs_i` is the S-wave velocity in km/s
- `rho_i` is the density in g/cm³
- `qp_i` is the P-wave quality factor
- `qs_i` is the S-wave quality factor

### Example

```
0.0 1.5 0.5 2.0 100 50
0.1 1.8 0.6 2.1 120 60
0.5 2.0 0.8 2.2 150 75
1.0 2.5 1.0 2.3 200 100
...
```

## Smoothing Data Format

Smoothing files define regions where velocity models should be smoothly transitioned between basins and the background model.

### Format Specification

```
num_points
lon_1 lat_1 weight_1
lon_2 lat_2 weight_2
...
lon_n lat_n weight_n
```

Where:
- `num_points` is the number of points in the smoothing boundary
- `lon_i` and `lat_i` are the longitude and latitude coordinates of point i
- `weight_i` is the smoothing weight at point i (typically between 0 and 1)

### Example

```
10
172.5000 -43.5000 0.0
172.5500 -43.5000 0.2
172.6000 -43.4800 0.5
172.6500 -43.4500 0.8
172.6500 -43.4000 1.0
172.6000 -43.3500 1.0
172.5500 -43.3200 0.8
172.5000 -43.3000 0.5
172.4500 -43.3200 0.2
172.4000 -43.3500 0.0
```

## Data Locations

The data files used by the NZCVM are located in the following directories:

- **Surface data**: `cvm/data/global/surface` and `cvm/data/regional/<basin_name>/`
- **Boundary data**: `cvm/data/regional/<basin_name>/`
- **Tomography data**: `cvm/data/global/tomography/`
- **1D velocity models**: `cvm/data/global/vm1d/`
- **Smoothing data**: `cvm/data/regional/<basin_name>/`
