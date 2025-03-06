"""
NaN Submodel Module.

This module provides functions to assign NaN values to velocities and density.
Used for locations where no valid velocity model can be applied.
"""

import numpy as np
from velocity_modelling.cvm.velocity3d import QualitiesVector


# def main(
#     zInd: int,
#     qualities_vector: QualitiesVector,
# ):
#     qualities_vector.rho[zInd] = np.nan
#     qualities_vector.vp[zInd] = np.nan
#     qualities_vector.vs[zInd] = np.nan


def main_vectorized(
    z_indices: np.ndarray,
    qualities_vector: QualitiesVector,
):
    """
    Assign NaN values to velocities and density at specified depth indices.

    Parameters
    ----------
    z_indices : np.ndarray
        Depth indices to assign NaN values to.
    qualities_vector : QualitiesVector
        Container for velocity and density data to be updated.
    """
    qualities_vector.vp[z_indices] = np.nan
    qualities_vector.vs[z_indices] = np.nan
    qualities_vector.rho[z_indices] = np.nan
