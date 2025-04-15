"""
NaN Submodel Module.

This module provides functions to assign NaN values to velocities and density.
Used for locations where no valid velocity model can be applied.
"""

import logging
from logging import Logger
from typing import Optional

import numpy as np

from velocity_modelling.velocity3d import (
    QualitiesVector,
)


def main_vectorized(
    z_indices: np.ndarray,
    qualities_vector: QualitiesVector,
    logger: Optional[Logger] = None,
):
    """
    Assign NaN values to velocities and density at specified depth indices.

    Parameters
    ----------
    z_indices : np.ndarray
        Depth indices to assign NaN values to.
    qualities_vector : QualitiesVector
        Container for velocity and density data to be updated.
    logger : Logger, optional
        Logger for reporting processing
    """
    if logger is not None:
        logger.log(
            logging.DEBUG,
            f"Assigning NaN layer properties to {len(z_indices)} points",
        )
    qualities_vector.vp[z_indices] = np.nan
    qualities_vector.vs[z_indices] = np.nan
    qualities_vector.rho[z_indices] = np.nan
