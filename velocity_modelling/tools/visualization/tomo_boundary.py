import h5py
import numpy as np
import geojson
import argparse
from pathlib import Path
from shapely.geometry import MultiPoint

def extract_single_boundary_geojson(hdf5_file_path: Path):
    with h5py.File(hdf5_file_path, 'r') as f:
        # Use the first available group to extract lat/lon
        first_group_name = next(iter(f.keys()))
        group = f[first_group_name]

        lats = group["latitudes"][:]
        lons = group["longitudes"][:]

        # Construct edge points only
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        edge_coords = []

        # Top and bottom rows
        edge_coords += list(zip(lon_grid[0, :], lat_grid[0, :]))          # top
        edge_coords += list(zip(lon_grid[-1, :], lat_grid[-1, :]))        # bottom

        # Left and right columns (excluding corners)
        edge_coords += list(zip(lon_grid[1:-1, 0], lat_grid[1:-1, 0]))     # left
        edge_coords += list(zip(lon_grid[1:-1, -1], lat_grid[1:-1, -1]))   # right

        # Compute convex hull
        hull = MultiPoint(edge_coords).convex_hull

        if hull.geom_type != 'Polygon':
            raise RuntimeError("Convex hull did not produce a polygon.")

        coords = [(x, y) for x, y in hull.exterior.coords]
        polygon = geojson.Polygon([coords])
        feature = geojson.Feature(geometry=polygon, properties={})
        feature_collection = geojson.FeatureCollection([feature])

        output_path = hdf5_file_path.with_suffix('.geojson')
        with open(output_path, 'w') as geojson_file:
            geojson.dump(feature_collection, geojson_file, indent=2)

        print(f"Boundary GeoJSON saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GeoJSON boundary from lat/lon grid in HDF5 file.")
    parser.add_argument("hdf5_file", type=str, help="Path to the HDF5 file (e.g., 2020_NZ.h5)")
    args = parser.parse_args()

    input_path = Path(args.hdf5_file).resolve()
    if not input_path.is_file():
        print(f"Error: File not found: {input_path}")
    else:
        extract_single_boundary_geojson(input_path)

