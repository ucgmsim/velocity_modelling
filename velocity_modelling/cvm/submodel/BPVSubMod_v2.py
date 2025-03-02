import numpy as np
from velocity_modelling.cvm.velocity3d import QualitiesVector


def main(
    zInd: int,
    qualities_vector: QualitiesVector,
):
    """
    Purpose:   calculate the rho vp and vs values at a single lat long depth point

    Input variables:
    zInd - the index of the grid point to store the data at
    qualities_vector - dict housing Vp, Vs, and Rho for one Lat Lon value and one or more depths

    Output variables:
    n.a.
    """
    qualities_vector.rho[zInd] = 2.334
    qualities_vector.vp[zInd] = 3.60
    qualities_vector.vs[zInd] = 1.9428


def main_vectorized(
    z_indices: np.ndarray,
    qualities_vector: QualitiesVector,
):
    """
    Purpose: Calculate the rho, vp, and vs values for multiple lat-long-depth points.

    Input variables:
    z_indices - array of indices of the grid points to store the data at
    qualities_vector - struct housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths

    Output variables: n.a.
    """
    qualities_vector.rho[z_indices] = 2.334
    qualities_vector.vp[z_indices] = 3.60
    qualities_vector.vs[z_indices] = 1.9428
