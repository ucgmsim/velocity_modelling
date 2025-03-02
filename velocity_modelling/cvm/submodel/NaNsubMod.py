import numpy as np
from velocity_modelling.cvm.velocity3d import QualitiesVector


def main(
    zInd: int,
    qualities_vector: QualitiesVector,
):
    qualities_vector.rho[zInd] = np.nan
    qualities_vector.vp[zInd] = np.nan
    qualities_vector.vs[zInd] = np.nan


def main_vectorized(
    z_indices: np.ndarray,
    qualities_vector: QualitiesVector,
):
    qualities_vector.rho[z_indices] = np.nan
    qualities_vector.vp[z_indices] = np.nan
    qualities_vector.vs[z_indices] = np.nan
