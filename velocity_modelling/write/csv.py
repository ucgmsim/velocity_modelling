"""
Module for writing velocity model data to CSV format.

"""

import logging
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from velocity_modelling.geometry import (
    PartialGlobalMesh,
)
from velocity_modelling.velocity3d import (
    PartialGlobalQualities,
)


def write_global_qualities(
    output_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities: PartialGlobalQualities,
    lat_ind: int,
    vm_params: dict,
    logger: Optional[Logger] = None,
):
    """
    Write the velocity model to CSV format using pandas.

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
    vm_params : dict
        Dictionary containing velocity model parameters from nzcvm.cfg.
    logger : Logger, optional
        Logger instance for logging messages.
    """
    if logger is None:
        logger = Logger("csv.wrote_global_qualities")

    min_vs = vm_params.get("min_vs", 0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "velocity_model.csv"

    mode = "w" if lat_ind == 0 else "a"
    header = lat_ind == 0

    if lat_ind == 0:
        logger.log(logging.DEBUG, f"Creating new CSV file: {output_file}")

    logger.log(logging.DEBUG, f"Writing CSV output for latitude index {lat_ind}")

    try:
        nx = partial_global_mesh.nx
        nz = partial_global_mesh.nz

        # Create data arrays
        data = {
            "y": np.full(nx * nz, lat_ind),
            "x": np.repeat(np.arange(nx), nz),
            "z": np.tile(np.arange(nz), nx),
            "lat": np.repeat(partial_global_mesh.lat, nz),
            "lon": np.repeat(partial_global_mesh.lon, nz),
            "depth": np.tile(
                partial_global_mesh.z * (-1), nx
            ),  # negative to convert elevation to depth
        }

        # Process quality values
        vp_flat = np.array(
            [partial_global_qualities.vp[k][i] for k in range(nx) for i in range(nz)]
        )
        vs_flat = np.array(
            [
                max(partial_global_qualities.vs[k][i], min_vs)
                for k in range(nx)
                for i in range(nz)
            ]
        )
        rho_flat = np.array(
            [partial_global_qualities.rho[k][i] for k in range(nx) for i in range(nz)]
        )
        inbasin_flat = np.array(
            [
                partial_global_qualities.inbasin[k][i]
                for k in range(nx)
                for i in range(nz)
            ]
        )

        data.update(
            {"vp": vp_flat, "vs": vs_flat, "rho": rho_flat, "inbasin": inbasin_flat}
        )

        # Create DataFrame and write to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, mode=mode, header=header, index=False)

    except Exception as e:
        logger.log(logging.ERROR, f"Error writing CSV data: {str(e)}")
        raise
