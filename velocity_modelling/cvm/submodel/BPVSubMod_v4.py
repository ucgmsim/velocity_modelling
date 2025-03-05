import numpy as np

from velocity_modelling.cvm.basin_model import PartialBasinSurfaceDepths
from velocity_modelling.cvm.global_model import PartialGlobalSurfaceDepths

from velocity_modelling.cvm.velocity3d import QualitiesVector
from velocity_modelling.cvm.gtl import v30gtl_vectorized
from velocity_modelling.cvm.submodel.BPVSubMod_v1 import vs_full, vp_full, rho_full


# def main(
#     zInd: int,
#     depth: float,
#     qualities_vector: QualitiesVector,
#     partial_basin_surface_depths: PartialBasinSurfaceDepths,
#     partial_global_surface_depths: PartialGlobalSurfaceDepths,
# ):
#     """
#     Purpose:   calculate the rho vp and vs values at a single lat long depth point
#
#     Input variables:
#     zInd - the index of the grid point to store the data at
#     depth - the depth of the grid point of interest. in meters. negative value
#     qualities_vector - dict housing Vp, Vs, and Rho for one Lat Lon value and one or more depths
#     partial_basin_surface_depths - dict containing the depth of the basin surface
#     partial_global_surface_depths - dict containing the depth of the global surface
#
#     Output variables:
#     n.a.
#     """
#
#     point_depth = partial_basin_surface_depths.depths[0] - depth
#
#     DEM_depth = partial_global_surface_depths.depths[1]  # value of the DEM
#     BPV_top = partial_basin_surface_depths.depths[0]  # value of the BPV top
#
#     z_DEM_relative = DEM_depth - depth  # delth of the gridpoint relative to the DEM
#     z_BPV_relative = BPV_top - depth  # delth of the gridpoint relative to the BPV top
#
#     ely_taper_depth = 350  # depth of the taper
#     vs30_taper_depth = 1000
#     vs0 = 0.700
#     vs_depth = 1.500
#     vs_ely_depth = 2.2818
#
#     if z_DEM_relative < vs30_taper_depth and z_BPV_relative < ely_taper_depth:
#         vs_BPV_top = (
#             vs0 + (vs_depth - vs0) * (z_DEM_relative / vs30_taper_depth)
#         ) * 1000  # convert to m/s
#         (
#             qualities_vector.vs[zInd],
#             qualities_vector.vp[zInd],
#             qualities_vector.rho[zInd],
#         ) = v30gtl(vs_BPV_top, vs_ely_depth, z_BPV_relative, ely_taper_depth)
#     else:
#
#         qualities_vector.rho[zInd] = rho_full
#         qualities_vector.vp[zInd] = vp_full
#         qualities_vector.vs[zInd] = vs_full


def main_vectorized(
    z_indices: np.ndarray,
    depths: np.ndarray,
    qualities_vector: QualitiesVector,
    partial_basin_surface_depths: PartialBasinSurfaceDepths,
    partial_global_surface_depths: PartialGlobalSurfaceDepths,
):
    """
    Purpose: Calculate the rho, vp, and vs values for multiple lat-long-depth points.

    Input variables:
    z_indices - array of indices of the grid points to store the data at
    depths - array of depths of the grid points of interest, in meters, negative values
    qualities_vector - struct housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths
    partial_basin_surface_depths - struct containing the depth of the basin surface
    partial_global_surface_depths - struct containing the depth of the global surface

    Output variables: n.a.
    """
    DEM_depth = partial_global_surface_depths.depths[1]  # value of the DEM
    BPV_top = partial_basin_surface_depths.depths[0]  # value of the BPV top

    z_DEM_relative = DEM_depth - depths  # Shape: (n,)
    z_BPV_relative = BPV_top - depths  # Shape: (n,)

    ely_taper_depth = 350  # depth of the taper
    vs30_taper_depth = 1000
    vs0 = 0.700
    vs_depth = 1.500
    vs_ely_depth = 2.2818

    # Vectorized condition
    gtl_mask = (z_DEM_relative < vs30_taper_depth) & (z_BPV_relative < ely_taper_depth)

    # Initialize with default full values
    qualities_vector.rho[z_indices] = rho_full
    qualities_vector.vp[z_indices] = vp_full
    qualities_vector.vs[z_indices] = vs_full

    # Apply GTL where the condition holds
    if np.any(gtl_mask):
        z_indices_gtl = z_indices[gtl_mask]
        z_DEM_relative_gtl = z_DEM_relative[gtl_mask]
        z_BPV_relative_gtl = z_BPV_relative[gtl_mask]

        vs_BPV_top = (
            vs0 + (vs_depth - vs0) * (z_DEM_relative_gtl / vs30_taper_depth)
        ) * 1000  # Convert to m/s

        vs_new, vp_new, rho_new = v30gtl_vectorized(
            vs_BPV_top, vs_ely_depth, z_BPV_relative_gtl, ely_taper_depth
        )
        qualities_vector.vs[z_indices_gtl] = vs_new
        qualities_vector.vp[z_indices_gtl] = vp_new
        qualities_vector.rho[z_indices_gtl] = rho_new
