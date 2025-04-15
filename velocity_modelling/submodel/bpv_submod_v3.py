"""
Banks Peninsula Volcanics (BPV) Velocity Submodel v3

This module implements a depth-dependent velocity model for the Banks Peninsula Volcanics area.
Unlike previous versions that used constant values, this model adjusts velocities based on depth,
providing a more realistic representation of volcanic rock properties.
"""

import logging
from logging import Logger
from typing import Optional

import numpy as np

from velocity_modelling.basin_model import (
    PartialBasinSurfaceDepths,
)
from velocity_modelling.interpolate import (
    linear_interpolation_vectorized,
)
from velocity_modelling.submodel.bpv_submod_v1 import (
    rho_full,
    vp_full,
    vs_full,
)
from velocity_modelling.velocity3d import (
    QualitiesVector,
)


def main_vectorized(
    z_indices: np.ndarray,
    depths: np.ndarray,
    qualities_vector: QualitiesVector,
    partial_basin_surface_depths: PartialBasinSurfaceDepths,
    logger: Optional[Logger] = None,
):
    """
    Calculate the rho, vp, and vs values for multiple lat-long-depth points.

    Parameters
    ----------
    z_indices : np.ndarray
        Array of indices of the grid points to store the data at.
    depths : np.ndarray
        Array of depths of the grid points of interest, in metres, negative values.
    qualities_vector : QualitiesVector
        Object housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths.
    partial_basin_surface_depths : PartialBasinSurfaceDepths
        Struct containing the depth of the basin surface.
    logger : Logger, optional
        Logger for reporting processing status.
    """

    if logger is not None:
        logger.log(
            logging.DEBUG,
            f"Assigning BPV v3 depth-dependent properties to {len(z_indices)} points",
        )

    weather_depth = 100
    point_depths = partial_basin_surface_depths.depths[0] - depths  # Shape: (n,)

    vs0 = 1.59  # vs at the top of the BPV
    vp0 = 3.2  # vp at the top of the BPV
    rho0 = 2.265  # rho at the top of the BPV

    # Vectorized condition
    weather_mask = point_depths < weather_depth

    # Initialize with default full values
    qualities_vector.rho[z_indices] = rho_full
    qualities_vector.vp[z_indices] = vp_full
    qualities_vector.vs[z_indices] = vs_full

    # Apply interpolation where point_depth < weather_depth
    if np.any(weather_mask):
        z_indices_weather = z_indices[weather_mask]
        point_depths_weather = point_depths[weather_mask]

        qualities_vector.rho[z_indices_weather] = linear_interpolation_vectorized(
            0, weather_depth, rho0, rho_full, point_depths_weather
        )
        qualities_vector.vp[z_indices_weather] = linear_interpolation_vectorized(
            0, weather_depth, vp0, vp_full, point_depths_weather
        )
        qualities_vector.vs[z_indices_weather] = linear_interpolation_vectorized(
            0, weather_depth, vs0, vs_full, point_depths_weather
        )
