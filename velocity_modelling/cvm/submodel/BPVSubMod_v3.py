import numpy as np

from velocity_modelling.cvm.basin_model import PartialBasinSurfaceDepths
from velocity_modelling.cvm.velocity3d import QualitiesVector
from velocity_modelling.cvm.interpolate import (
    linear_interpolation_vectorized,
)
from velocity_modelling.cvm.submodel.BPVSubMod_v1 import vs_full, vp_full, rho_full


# def main(
#     zInd: int,
#     depth: float,
#     qualities_vector: QualitiesVector,
#     partial_basin_surface_depths: PartialBasinSurfaceDepths,
# ):
#     """
#     Purpose:   calculate the rho vp and vs values at a single lat long depth point
#
#     Input variables:
#     zInd - the index of the grid point to store the data at
#     dep - the depth of the grid point of interest. in meters. negative value
#     qualities_vector - dict housing Vp, Vs, and Rho for one Lat Lon value and one or more depths
#     velo_mod_1d_data - dict containing a 1D velocity model
#
#     Output variables:
#     n.a.
#     """
#     weather_depth = 100
#     point_depth = partial_basin_surface_depths.depth[0] - depth
#
#     vs0 = 1.59  # vs at the top of the BPV
#     vp0 = 3.2  # vp at the top of the BPV
#     rho0 = 2.265  # rho at the top of the BPV
#
#     if point_depth < weather_depth:
#         qualities_vector.rho[zInd] = linear_interpolation(
#             0, weather_depth, rho0, rho_full, point_depth
#         )
#         qualities_vector.vp[zInd] = linear_interpolation(
#             0, weather_depth, vp0, vp_full, point_depth
#         )
#         qualities_vector.vs[zInd] = linear_interpolation(
#             0, weather_depth, vs0, vs_full, point_depth
#         )
#     else:
#         qualities_vector.rho[zInd] = rho_full
#         qualities_vector.vp[zInd] = vp_full
#         qualities_vector.vs[zInd] = vs_full


def main_vectorized(
    z_indices: np.ndarray,
    depths: np.ndarray,
    qualities_vector: QualitiesVector,
    partial_basin_surface_depths: PartialBasinSurfaceDepths,
):
    """
    Purpose: Calculate the rho, vp, and vs values for multiple lat-long-depth points.

    Input variables:
    z_indices - array of indices of the grid points to store the data at
    depths - array of depths of the grid points of interest, in meters, negative values
    qualities_vector - struct housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths
    partial_basin_surface_depths - struct containing the depth of the basin surface

    Output variables: n.a.
    """
    weather_depth = 100
    point_depths = partial_basin_surface_depths.depths[0] - depths  # Shape: (n,)

    vs0 = 1.59  # vs at the top of the BPV
    vp0 = 3.2  # vp at the top of the BPV
    rho0 = 2.265  # rho at the top of the BPV

    # Vectorized condition
    weather_mask = point_depths < weather_depth

    # Initialize with default full values
    qualities_vector.rho[z_indices] = rho_full
    qualities_vector.vp[z_indices] = vp_full
    qualities_vector.vs[z_indices] = vs_full

    # Apply interpolation where point_depth < weather_depth
    if np.any(weather_mask):
        z_indices_weather = z_indices[weather_mask]
        point_depths_weather = point_depths[weather_mask]

        qualities_vector.rho[z_indices_weather] = linear_interpolation_vectorized(
            0, weather_depth, rho0, rho_full, point_depths_weather
        )
        qualities_vector.vp[z_indices_weather] = linear_interpolation_vectorized(
            0, weather_depth, vp0, vp_full, point_depths_weather
        )
        qualities_vector.vs[z_indices_weather] = linear_interpolation_vectorized(
            0, weather_depth, vs0, vs_full, point_depths_weather
        )
