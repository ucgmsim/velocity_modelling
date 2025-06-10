"""
Canterbury 1D Velocity Model v1

This module provides a one-dimensional velocity model for the Canterbury region.
It assigns P-wave velocity, S-wave velocity, and density values based on depth
using a predefined 1D velocity profile.
"""

import logging
from logging import Logger
from typing import Optional

import numpy as np

from velocity_modelling.velocity1d import (
    VelocityModel1D,
)
from velocity_modelling.velocity3d import (
    QualitiesVector,
)


def main_vectorized(
    z_indices: np.ndarray,
    depths: np.ndarray,
    qualities_vector: QualitiesVector,
    vm1d_data: VelocityModel1D,
    logger: Optional[Logger] = None,
):
    """
    Calculate rho, vp, and vs values for multiple lat-long-depth points.

    Parameters
    ----------
    z_indices : np.ndarray
        Array of indices of the grid points to store the data at.
    depths : np.ndarray
        Array of depths of the grid points of interest, in metres, negative values.
    qualities_vector : QualitiesVector
        Object housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths.
    vm1d_data : VelocityModel1D
        Object containing a 1D velocity model.
    logger : Logger, optional
        Logger for reporting processing status.
    """
    if logger is not None:
        logger.log(
            logging.DEBUG, f"Applying Canterbury 1D model to {len(z_indices)} points"
        )

    # Convert model depths to meters, negative downwards
    model_depths = (
        np.array(vm1d_data.bottom_depth) * -1000
    )  # Shape: (num_model_depths,)

    # Define tolerance for floating-point comparisons
    tolerance = 1e-6

    # Create ascending version of model_depths for searchsorted
    model_depths_asc = model_depths[
        ::-1
    ]  # e.g., [..., -800, -600, -400, -300, -200, -150, -100, -50]

    # Find first index where model_depth <= depth + tolerance
    indices = len(model_depths) - np.searchsorted(
        model_depths_asc, depths + tolerance, side="right"
    )

    # Mask for valid indices (within model extent)
    valid_mask = (indices >= 0) & (indices < len(model_depths))

    if not np.all(valid_mask):
        invalid_indices = np.where(~valid_mask)[0]
        for idx in invalid_indices:
            logger.log(
                logging.ERROR,
                f"Depth point {depths[idx]} below the extent represented in the 1D velocity model file.",
            )

    # Apply valid mask to indices and z_indices
    valid_indices = indices[valid_mask]
    valid_z_indices = z_indices[valid_mask]

    # Assign values for valid indices
    if valid_indices.size > 0:
        qualities_vector.rho[valid_z_indices] = np.array(vm1d_data.rho)[valid_indices]
        qualities_vector.vp[valid_z_indices] = np.array(vm1d_data.vp)[valid_indices]
        qualities_vector.vs[valid_z_indices] = np.array(vm1d_data.vs)[valid_indices]
