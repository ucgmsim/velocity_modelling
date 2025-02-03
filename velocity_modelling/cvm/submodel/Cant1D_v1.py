import numpy as np
from velocity_modelling.cvm.registry import QualitiesVector


def main(
    zInd: int,
    dep: float,
    qualities_vector: QualitiesVector,
    VELO_MOD_1D_DATA: Dict[str, np.ndarray],
):
    """
    Purpose:   calculate the rho vp and vs values at a single lat long depth point

    Input variables:
    zInd - the indicie of the grid point to store the data at
    dep - the depth of the grid point of interest
    QUALITIES_VECTOR - dict housing Vp, Vs, and Rho for one Lat Lon value and one or more depths
    VELO_MOD_1D_DATA - dict containing a 1D velocity model

    Output variables:
    n.a.
    """
    depths = np.array(VELO_MOD_1D_DATA["Dep"]) * -1000
    idx = np.searchsorted(depths, dep, side="right") - 1

    if idx >= 0:
        qualities_vector.Rho[zInd] = VELO_MOD_1D_DATA["Rho"][idx]
        qualities_vector.Vp[zInd] = VELO_MOD_1D_DATA["Vp"][idx]
        qualities_vector.Vs[zInd] = VELO_MOD_1D_DATA["Vs"][idx]
    else:
        print(
            "Error: Depth point below the extent represented in the 1D velocity model file."
        )
