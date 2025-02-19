from velocity_modelling.cvm.registry import QualitiesVector

vs_full = 2.2818  # vs at the full
vp_full = 4.0  # vp at the full
rho_full = 2.393  # rho at the full


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
    qualities_vector.rho[zInd] = rho_full
    qualities_vector.vp[zInd] = vp_full
    qualities_vector.vs[zInd] = vs_full
