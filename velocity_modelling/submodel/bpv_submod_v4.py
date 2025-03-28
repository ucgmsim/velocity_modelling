"""
Banks Peninsula Volcanics (BPV) Velocity Submodel v4

This module implements an advanced velocity model for the Banks Peninsula Volcanics area.
It provides depth-dependent velocity values for the BPV region, adjusting properties based on depth
to better represent the geology of the area. The model includes a GTL component for improved accuracy.

"""

import logging
from logging import Logger
from typing import Optional

import numpy as np

from velocity_modelling.basin_model import (
    PartialBasinSurfaceDepths,
)
from velocity_modelling.global_model import (
    PartialGlobalSurfaceDepths,
)
from velocity_modelling.gtl import v30gtl_vectorized
from velocity_modelling.submodel.bpv_submod_v1 import (
    rho_full,
    vp_full,
    vs_full,
)
from velocity_modelling.velocity3d import (
    QualitiesVector,
)


def main_vectorized(
    z_indices: np.ndarray,
    depths: np.ndarray,
    qualities_vector: QualitiesVector,
    partial_basin_surface_depths: PartialBasinSurfaceDepths,
    partial_global_surface_depths: PartialGlobalSurfaceDepths,
    logger: Optional[Logger] = None,
):
    """
    Calculate the rho, vp, and vs values for multiple lat-long-depth points.

    Parameters
    ----------
    z_indices : np.ndarray
        Array of indices of the grid points to store the data at.
    depths : np.ndarray
        Array of depths of the grid points of interest, in metres, negative values.
    qualities_vector : QualitiesVector
        Object housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths.
    partial_basin_surface_depths : PartialBasinSurfaceDepths
        Struct containing the depth of the basin surface.
    partial_global_surface_depths : PartialGlobalSurfaceDepths
        Struct containing the depth of the global surface.
    logger : Logger, optional
        Logger for reporting processing status.

    """
    if logger is not None:
        logger.log(
            logging.DEBUG,
            f"Assigning BPV v4 depth-dependent properties to {len(z_indices)} points",
        )

    dem_depth = partial_global_surface_depths.depths[1]  # value of the DEM
    bpv_top = partial_basin_surface_depths.depths[0]  # value of the BPV top

    z_dem_relative = dem_depth - depths  # Shape: (n,)
    z_bpv_relative = bpv_top - depths  # Shape: (n,)

    ely_taper_depth = 350  # depth of the taper
    vs30_taper_depth = 1000
    vs0 = 0.700
    vs_depth = 1.500
    vs_ely_depth = 2.2818

    # Vectorized condition
    gtl_mask = (z_dem_relative < vs30_taper_depth) & (z_bpv_relative < ely_taper_depth)

    # Initialize with default full values
    qualities_vector.rho[z_indices] = rho_full
    qualities_vector.vp[z_indices] = vp_full
    qualities_vector.vs[z_indices] = vs_full

    # Apply GTL where the condition holds
    if np.any(gtl_mask):
        z_indices_gtl = z_indices[gtl_mask]
        z_dem_relative_gtl = z_dem_relative[gtl_mask]
        z_bpv_relative_gtl = z_bpv_relative[gtl_mask]

        vs_bpv_top = (
            vs0 + (vs_depth - vs0) * (z_dem_relative_gtl / vs30_taper_depth)
        ) * 1000  # Convert to m/s

        vs_new, vp_new, rho_new = v30gtl_vectorized(
            vs_bpv_top, vs_ely_depth, z_bpv_relative_gtl, ely_taper_depth
        )
        qualities_vector.vs[z_indices_gtl] = vs_new
        qualities_vector.vp[z_indices_gtl] = vp_new
        qualities_vector.rho[z_indices_gtl] = rho_new
