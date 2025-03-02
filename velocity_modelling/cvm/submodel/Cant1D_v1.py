import numpy as np
from velocity_modelling.cvm.velocity1d import VelocityModel1D
from velocity_modelling.cvm.velocity3d import QualitiesVector


def main(
    zInd: int,
    depth: float,
    qualities_vector: QualitiesVector,
    velo_mod_1d_data: VelocityModel1D,
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
        np.array(velo_mod_1d_data.depth) * -1000
    )  # convert to meters. negative being downwards
    idx = len(depths) - np.searchsorted(
        depths[::-1], depth, side="right"
    )  # depths are in descending order, so reverse the array

    if idx >= 0:
        qualities_vector.rho[zInd] = velo_mod_1d_data.rho[idx]
        qualities_vector.vp[zInd] = velo_mod_1d_data.vp[idx]
        qualities_vector.vs[zInd] = velo_mod_1d_data.vs[idx]
    else:
        print(
            "Error: Depth point below the extent represented in the 1D velocity model file."
        )


def main_vectorized(
    z_indices: np.ndarray,
    depths: np.ndarray,
    qualities_vector: QualitiesVector,
    velo_mod_1d_data: VelocityModel1D,
):
    """
    Purpose: Calculate rho, vp, and vs values for multiple lat-long-depth points.

    Input variables:
    z_indices - array of indices of the grid points to store the data at
    depths - array of depths of the grid points of interest, in meters, negative values
    qualities_vector - struct housing Vp, Vs, and Rho for one Lat-Lon value and multiple depths
    velo_mod_1d_data - struct containing a 1D velocity model

    Output variables: n.a.
    """
    # Convert model depths to meters, negative downwards
    model_depths = (
        np.array(velo_mod_1d_data.depth) * -1000
    )  # Shape: (num_model_depths,)

    # Vectorized search for indices
    # model_depths is in descending order, so reverse for searchsorted
    indices = len(model_depths) - np.searchsorted(
        model_depths[::-1], depths, side="right"
    )

    # Mask for valid indices (depths within model extent)
    valid_mask = indices >= 0

    if not np.all(valid_mask):
        print(
            "Error: Some depth points are below the extent represented in the 1D velocity model file."
        )

    # Apply valid mask to indices and z_indices
    valid_indices = indices[valid_mask]
    valid_z_indices = z_indices[valid_mask]

    # Assign values for valid indices
    if valid_indices.size > 0:
        qualities_vector.rho[valid_z_indices] = np.array(velo_mod_1d_data.rho)[
            valid_indices
        ]
        qualities_vector.vp[valid_z_indices] = np.array(velo_mod_1d_data.vp)[
            valid_indices
        ]
        qualities_vector.vs[valid_z_indices] = np.array(velo_mod_1d_data.vs)[
            valid_indices
        ]
