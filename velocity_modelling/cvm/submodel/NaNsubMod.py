import numpy as np
from velocity_modelling.cvm.registry import QualitiesVector, VeloMod1DData


def main(
    zInd: int,
    qualities_vector: QualitiesVector,
):
    qualities_vector.rho[zInd] = np.nan
    qualities_vector.vp[zInd] = np.nan
    qualities_vector.vs[zInd] = np.nan
