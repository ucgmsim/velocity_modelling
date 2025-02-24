import numpy as np
from velocity_modelling.cvm.velocity import QualitiesVector


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
    qualities_vector.rho[zInd] = 2.151
    qualities_vector.vp[zInd] = 2.7
    qualities_vector.vs[zInd] = 1.1511
