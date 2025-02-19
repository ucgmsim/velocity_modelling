import numpy as np
from velocity_modelling.cvm.registry import QualitiesVector, VeloMod1DData


def main(
    zInd: int,
    dep: float,
    qualities_vector: QualitiesVector,
    velo_mod_1d_data: VeloMod1DData,
):
    """
    Purpose:   calculate the rho vp and vs values at a single lat long depth point

    Input variables:
    zInd - the index of the grid point to store the data at
    dep - the depth of the grid point of interest. in meters. negative value
    qualities_vector - dict housing Vp, Vs, and Rho for one Lat Lon value and one or more depths
    velo_mod_1d_data - dict containing a 1D velocity model

    Output variables:
    n.a.
    """
    depths = (
        np.array(velo_mod_1d_data.dep) * -1000
    )  # convert to meters. negative being downwards
    idx = len(depths) - np.searchsorted(
        depths[::-1], dep, side="right"
    )  # depths are in decending order, so reverse the array

    if idx >= 0:
        qualities_vector.rho[zInd] = velo_mod_1d_data.rho[idx]
        qualities_vector.vp[zInd] = velo_mod_1d_data.vp[idx]
        qualities_vector.vs[zInd] = velo_mod_1d_data.vs[idx]
    else:
        print(
            "Error: Depth point below the extent represented in the 1D velocity model file."
        )
