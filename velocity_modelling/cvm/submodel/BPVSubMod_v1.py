"""
Banks Peninsula Volcanics (BPV) Velocity Submodel v1

This module provides constant velocity values for the Banks Peninsula Volcanics area.
It implements a simple model with fixed P-wave velocity, S-wave velocity, and density
values for all points within the BPV boundaries.
"""

import numpy as np
from velocity_modelling.cvm.velocity3d import QualitiesVector
from velocity_modelling.cvm.logging import VMLogger


# Constants
vs_full = 2.2818  # vs at the full (km/s)
vp_full = 4.0  # vp at the full (km/s)
rho_full = 2.393  # rho at the full (g/cm³)


def main_vectorized(
    z_indices: np.ndarray,
    qualities_vector: QualitiesVector,
    logger: VMLogger = None,
):
    """
    Calculate rho, vp, and vs values for multiple lat-long-depth points.

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
            f"Assigning BPV v1 properties to {len(z_indices)} points", logger.DEBUG
        )

    qualities_vector.rho[z_indices] = rho_full
    qualities_vector.vp[z_indices] = vp_full
    qualities_vector.vs[z_indices] = vs_full
