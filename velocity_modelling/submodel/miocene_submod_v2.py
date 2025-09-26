"""
Miocene Layer Velocity Submodel v2

This module provides constant velocity values for the Miocene geological layer.
The assigned values are based on empirical relationships for sedimentary rocks
of Miocene age in the Canterbury basin.
"""

import logging
from logging import Logger
from typing import Optional

import numpy as np

from velocity_modelling.velocity3d import (
    QualitiesVector,
)


def main_vectorized(
    z_indices: np.ndarray,
    qualities_vector: QualitiesVector,
    logger: Optional[Logger] = None,
):
    """
    Calculate rho, vp, and vs values for multiple lat-long-depth points in the Miocene layer.

    Parameters
    ----------
    z_indices : np.ndarray
        Array of indices of the grid points to store the data at.
    qualities_vector : QualitiesVector
        Object housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths.
    logger : Logger, optional
        Logger for reporting processing status.
    """
    if logger is not None:
        logger.log(
            logging.DEBUG,
            f"Assigning Miocene layer properties to {len(z_indices)} points",
        )
    qualities_vector.rho[z_indices] = 2.09
    qualities_vector.vp[z_indices] = 2.5
    qualities_vector.vs[z_indices] = 0.984
