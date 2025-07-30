"""
This script extracts the geographical boundary from a tomography HDF5 file
and saves it as a GeoJSON file. It computes the convex hull of the grid points
defined by the latitude and longitude arrays in the HDF5 file to determine
the boundary polygon.

The script is intended to be run from the command line, taking the path to an
HDF5 file as input.
"""

from pathlib import Path
from typing import Annotated

import geojson
import h5py
import numpy as np
import typer
from shapely.geometry import MultiPoint

from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


@cli.from_docstring(app)
def extract_single_boundary_geojson(
    hdf5_file_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the HDF5 file (e.g., 2020_NZ.h5)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
):
    """Extracts the convex hull boundary from an HDF5 file and saves it as GeoJSON.

    This function reads latitude and longitude grids from the first group in the
    HDF5 file, calculates the convex hull of the grid's edge points, and
    writes the resulting polygon to a GeoJSON file. The output file will have
    the same name as the input file but with a .geojson extension.

    Parameters
    ----------
    hdf5_file_path : Path
        The path to the input HDF5 file containing latitude and longitude grids.

    Raises
    ------
    RuntimeError
        If the convex hull operation does not result in a Polygon.
    """
    with h5py.File(hdf5_file_path, "r") as f:
        # Use the first available group to extract lat/lon
        first_group_name = next(iter(f.keys()))
        group = f[first_group_name]

        lats = group["latitudes"][:]
        lons = group["longitudes"][:]

        # Construct edge points only
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        edge_coords = []

        # Top and bottom rows
        edge_coords += list(zip(lon_grid[0, :], lat_grid[0, :]))  # top
        edge_coords += list(zip(lon_grid[-1, :], lat_grid[-1, :]))  # bottom

        # Left and right columns (excluding corners)
        edge_coords += list(zip(lon_grid[1:-1, 0], lat_grid[1:-1, 0]))  # left
        edge_coords += list(zip(lon_grid[1:-1, -1], lat_grid[1:-1, -1]))  # right

        # Compute convex hull
        hull = MultiPoint(edge_coords).convex_hull

        if hull.geom_type != "Polygon":
            raise RuntimeError("Convex hull did not produce a polygon.")

        coords = [(x, y) for x, y in hull.exterior.coords]
        polygon = geojson.Polygon([coords])
        feature = geojson.Feature(geometry=polygon, properties={})
        feature_collection = geojson.FeatureCollection([feature])

        output_path = hdf5_file_path.with_suffix(".geojson")
        with open(output_path, "w") as geojson_file:
            geojson.dump(feature_collection, geojson_file, indent=2)

        print(f"Boundary GeoJSON saved to: {output_path}")


if __name__ == "__main__":
    app()
