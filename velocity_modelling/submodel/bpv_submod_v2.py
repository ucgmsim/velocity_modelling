"""
Banks Peninsula Volcanics (BPV) Velocity Submodel v2

This module implements an updated velocity model for the Banks Peninsula Volcanics area.
It applies modified velocity values compared to v1, with improved constants based on
more recent geophysical measurements.
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
    Calculate the rho, vp, and vs values for multiple lat-long-depth points.

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
            logging.DEBUG, f"Assigning BPV v2 properties to {len(z_indices)} points"
        )

    qualities_vector.rho[z_indices] = 2.334
    qualities_vector.vp[z_indices] = 3.60
    qualities_vector.vs[z_indices] = 1.9428
