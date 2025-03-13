"""
Module for writing velocity model data to CSV format.
"""

import csv
import logging
from logging import Logger
from pathlib import Path

from velocity_modelling.cvm.geometry import (
    PartialGlobalMesh,
)
from velocity_modelling.cvm.velocity3d import (
    PartialGlobalQualities,
)


def write_global_qualities(
    output_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities: PartialGlobalQualities,
    lat_ind: int,
    min_vs: float = 0.0,
    logger: Logger = None,
):
    """
    Write the velocity model to CSV format.

    Parameters
    ----------
    output_dir : Path
        Directory where the output files will be written.
    partial_global_mesh : PartialGlobalMesh
        Structure containing the latitude and longitude grid.
    partial_global_qualities : PartialGlobalQualities
        Structure containing Vp, Vs, and Rho for all grid points.
    lat_ind : int
        Latitude index to determine the write mode (write or append).
    min_vs : float, optional
        Minimum Vs value to apply to the model.
    logger : Logger
        Logger instance for logging messages.

    """
    if logger is None:
        logger = Logger("csv.wrote_global_qualities")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "velocity_model.csv"

    mode = "w" if lat_ind == 0 else "a"

    if lat_ind == 0:
        logger.log(logging.INFO, f"Creating new CSV file: {output_file}")

    logger.log(logging.DEBUG, f"Writing CSV output for latitude index {lat_ind}")

    try:
        with open(output_file, mode, newline="") as csvfile:
            writer = csv.writer(csvfile)

            if lat_ind == 0:
                writer.writerow(
                    ["y", "x", "z", "lat", "lon", "depth", "vp", "vs", "rho", "inbasin"]
                )

            # Write data for this latitude slice
            for k in range(partial_global_mesh.nx):  # Loop through longitude points
                current_lon = partial_global_mesh.lon[k]
                current_lat = partial_global_mesh.lat[
                    k
                ]  # Index with k only, since lat is 1D of length nx

                for i in range(partial_global_mesh.nz):  # Loop through depth points
                    # Get indices for grid position
                    y_index = lat_ind
                    x_index = k
                    z_index = i
                    depth = partial_global_mesh.z[i]

                    # Get model values
                    vp = partial_global_qualities.vp[k][i]
                    vs = max(partial_global_qualities.vs[k][i], min_vs)
                    rho = partial_global_qualities.rho[k][i]
                    inbasin = partial_global_qualities.inbasin[k][i]

                    # Write row
                    writer.writerow(
                        [
                            y_index,
                            x_index,
                            z_index,
                            current_lat,
                            current_lon,
                            depth,
                            vp,
                            vs,
                            rho,
                            inbasin,
                        ]
                    )

    except Exception as e:
        logger.log(logging.ERROR, f"Error writing CSV data: {str(e)}")
        raise
