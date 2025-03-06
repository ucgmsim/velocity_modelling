"""
Paleogene Layer Velocity Submodel v1

This module provides velocity values for the Paleogene geological layer.
It implements depth-dependent velocity values based on empirical relationships
for sedimentary rocks of Paleogene age in the Canterbury basin.
"""

import numpy as np

from velocity_modelling.cvm.logging import VMLogger
from velocity_modelling.cvm.velocity3d import (
    QualitiesVector,
)


def main_vectorized(
    z_indices: np.ndarray,
    qualities_vector: QualitiesVector,
    logger: VMLogger = None,
):
    """
    Calculate rho, vp, and vs values for multiple lat-long-depth points in the Paleogene layer.

    Parameters
    ----------
    z_indices : np.ndarray
        Array of indices of the grid points to store the data at.
    qualities_vector : QualitiesVector
        Object housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths.
    logger : VMLogger, optional
        Logger for reporting processing status.
    """

    qualities_vector.rho[z_indices] = 2.151
    qualities_vector.vp[z_indices] = 2.7
    qualities_vector.vs[z_indices] = 1.1511
