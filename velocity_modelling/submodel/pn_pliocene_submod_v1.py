"""
Pliocene Layer Velocity Submodel v1

This module provides constant velocity values for the Pliocene geological layer.
It assigns constant values for sedimentary rocks of Pliocene age in the Palmerston North region.
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
    Calculate rho, vp, and vs values for multiple lat-long-depth points in the Pliocene layer.

    Parameters
    ----------
    z_indices : np.ndarray
        Array of indices of the grid points to store the data at.
    qualities_vector : QualitiesVector
        Object housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths.
    logger : Logger, optional
        Logger for reporting processing
    """
    if logger is not None:
        logger.log(
            logging.DEBUG,
            f"Assigning Pliocene layer properties to {len(z_indices)} points",
        )

    qualities_vector.rho[z_indices] = 2.12
    qualities_vector.vp[z_indices] = 2.6
    qualities_vector.vs[z_indices] = 1.1
