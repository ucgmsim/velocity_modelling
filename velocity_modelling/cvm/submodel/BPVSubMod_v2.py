"""
Banks Peninsula Volcanics (BPV) Velocity Submodel v2

This module implements an updated velocity model for the Banks Peninsula Volcanics area.
It applies modified velocity values compared to v1, with improved constants based on
more recent geophysical measurements.
"""

import numpy as np
from velocity_modelling.cvm.velocity3d import QualitiesVector
from velocity_modelling.cvm.logging import VMLogger


def main_vectorized(
    z_indices: np.ndarray,
    qualities_vector: QualitiesVector,
    logger: VMLogger = None,
):
    """
    Calculate the rho, vp, and vs values for multiple lat-long-depth points.

    Parameters
    ----------
    z_indices : np.ndarray
        Array of indices of the grid points to store the data at.
    qualities_vector : QualitiesVector
        Object housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths.
    logger : VMLogger, optional
        Logger for reporting processing status.
    """

    if logger is not None:
        logger.log(
            f"Assigning BPV v2 properties to {len(z_indices)} points", logger.DEBUG
        )

    qualities_vector.rho[z_indices] = 2.334
    qualities_vector.vp[z_indices] = 3.60
    qualities_vector.vs[z_indices] = 1.9428
