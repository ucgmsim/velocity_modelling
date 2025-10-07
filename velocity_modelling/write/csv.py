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

    Raises
    ------
    ValueError
        If any velocity/density values are negative.
    OSError
        If file operations fail.
    """
    if logger is None:
        logger = Logger("csv.wrote_global_qualities")

    min_vs = vm_params.get("min_vs", 0.0)

    # Validate that all velocity/density values are non-negative
    vp_data = partial_global_qualities.vp
    vs_data = partial_global_qualities.vs
    rho_data = partial_global_qualities.rho

    if np.any(vp_data < 0):
        error_msg = f"Negative values found in Vp data at slice {lat_ind}. Min value: {np.min(vp_data)}"
        logger.log(logging.ERROR, error_msg)
        raise ValueError(error_msg)

    if np.any(vs_data < 0):
        error_msg = f"Negative values found in Vs data at slice {lat_ind}. Min value: {np.min(vs_data)}"
        logger.log(logging.ERROR, error_msg)
        raise ValueError(error_msg)

    if np.any(rho_data < 0):
        error_msg = f"Negative values found in density data at slice {lat_ind}. Min value: {np.min(rho_data)}"
        logger.log(logging.ERROR, error_msg)
        raise ValueError(error_msg)

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

        # Process quality values using efficient numpy operations
        # Apply min_vs constraint to vs_data before flattening
        vs_constrained = np.maximum(vs_data, min_vs)

        # Flatten arrays using transpose and flatten for consistent ordering
        vp_flat = vp_data.T.flatten()
        vs_flat = vs_constrained.T.flatten()
        rho_flat = rho_data.T.flatten()
        inbasin_flat = partial_global_qualities.inbasin.T.flatten()

        data.update(
            {"vp": vp_flat, "vs": vs_flat, "rho": rho_flat, "inbasin": inbasin_flat}
        )

        # Create DataFrame and write to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, mode=mode, header=header, index=False)

    except Exception as e:
        logger.log(logging.ERROR, f"Error writing CSV data: {str(e)}")
        raise
