"""
Eberhart-Phillips et al. (2010) Tomography Model

This module implements a 3D velocity model based on the tomography study by
Eberhart-Phillips et al. (2010) for New Zealand. It provides location-dependent
P-wave velocity, S-wave velocity, and density values derived from tomographic
inversions, and it also handles a range of data corrections.

Key features of this submodel include:
- **Depth-based Interpolation**: Calculates velocity values at specified depths by
  interpolating between the discrete elevation layers defined in the tomography model.
- **Geotechnical Layer (GTL) Correction**: Applies a near-surface velocity adjustment
  using the Ely (2010) GTL model, based on Vs30 values from a global surface.
- **Special Offshore Tapering**: A unique tapering feature that applies a
  separate 1D velocity model in specific offshore locations (Vs30 < 100 m/s, indicating soft sediments),
  to ensure a smooth and geologically sound transition from land to sea.

This module is designed to be called by the main velocity calculation module
(e.g., `velocity3d.py`) and is configured via the `nzcvm_registry.yaml` file.

"""

import logging
from logging import Logger
from typing import Optional

import numpy as np

from velocity_modelling.constants import VelocityTypes
from velocity_modelling.geometry import MeshVector
from velocity_modelling.global_model import (
    PartialGlobalSurfaceDepths,
    TomographyData,
)
from velocity_modelling.gtl import v30gtl_vectorized
from velocity_modelling.interpolate import (
    linear_interpolation_vectorized,
)
from velocity_modelling.submodel import canterbury1d_submod
from velocity_modelling.velocity3d import (
    QualitiesVector,
)


def offshore_basin_depth_vectorized(shoreline_dist: np.ndarray):
    """
    Calculate the offshore basin depth based on the distance from the shoreline.

    Parameters
    ----------
    shoreline_dist : np.ndarray
        Array of distances from the shoreline.

    Returns
    -------
    np.ndarray
        Array of basin depths.
    """
    return np.where(
        shoreline_dist > 50,
        -3000.0,
        np.where(
            shoreline_dist > 20,
            -2000.0 - (((shoreline_dist - 20) / 30.0) * 1000.0),
            (shoreline_dist / 20.0) * -2000.0,
        ),
    )


# TODO: Utilize DEFAULT_OFFSHORE_1D_MODEL and make this function more generic
def offshore_basinmodel_vectorized(
    distance_from_shoreline: np.ndarray,
    depths: np.ndarray,
    qualities_vector: QualitiesVector,
    z_indices: np.ndarray,
    nz_tomography_data: TomographyData,
):
    """
    Calculate the rho, vp, and vs values for multiple lat-long-depth points within this velocity submodel.


    Parameters
    ----------
    distance_from_shoreline : np.ndarray
        Array of distances from the shoreline.
    depths : np.ndarray
        Array of depth values.
    qualities_vector : QualitiesVector
        Struct containing vp, vs, and rho values.
    z_indices : np.ndarray
        Array of indices of the depth points.
    nz_tomography_data : TomographyData
        Struct containing New Zealand tomography data.
    """
    offshore_depths = offshore_basin_depth_vectorized(
        distance_from_shoreline
    )  # calculate offshore basin depths based on distance from shoreline
    offshore_apply_mask = (
        offshore_depths < depths
    )  # identify points where the actual depth values (depths) are greater than the calculated offshore basin depths
    z_indices_offshore = z_indices[offshore_apply_mask]
    depths_offshore = depths[offshore_apply_mask]
    if z_indices_offshore.size > 0:
        canterbury1d_submod.main_vectorized(
            z_indices_offshore,
            depths_offshore,
            qualities_vector,
            nz_tomography_data.offshore_basin_model_1d,
        )  # apply the Canterbury 1D model (=DEFAULT_OFFSHORE_1D_MODEL) to the offshore points


def _apply_gtl(
    z_indices: np.ndarray,
    relative_depths: np.ndarray,
    qualities_vector: QualitiesVector,
    mesh_vector: MeshVector,
):
    """
    Helper function to apply the GTL model to a set of points.

    Parameters
    ----------
    z_indices : np.ndarray
        Array of indices of the depth points.
    relative_depths : np.ndarray
        Array of relative depth values.
    qualities_vector : QualitiesVector
        Struct containing vp, vs, and rho values.
    mesh_vector : MeshVector
        Struct containing mesh information such as latitude, longitude, and vs30.

    """
    gtl_mask = relative_depths <= 350.0
    if np.any(gtl_mask):
        z_indices_gtl = z_indices[gtl_mask]
        vs_gtl = qualities_vector.vs[z_indices_gtl]
        relative_depths_gtl = relative_depths[gtl_mask]
        vs_new, vp_new, rho_new = v30gtl_vectorized(
            mesh_vector.vs30, vs_gtl, relative_depths_gtl, 350.0
        )
        qualities_vector.vs[z_indices_gtl] = vs_new
        qualities_vector.vp[z_indices_gtl] = vp_new
        qualities_vector.rho[z_indices_gtl] = rho_new


def main_vectorized(
    z_indices: np.ndarray,
    depths: np.ndarray,
    qualities_vector: QualitiesVector,
    mesh_vector: MeshVector,
    nz_tomography_data: TomographyData,
    partial_global_surface_depths: PartialGlobalSurfaceDepths,
    in_any_basin_lat_lon: bool,
    on_boundary: bool,
    interpolated_global_surface_values: dict,
    logger: Optional[Logger] = None,
):
    """
    Calculate rho, vp, and vs values for multiple lat-long-depth points using the
    Eberhart-Phillips et al. (2010) tomography model.

    Parameters
    ----------
    z_indices : np.ndarray
        Array of indices of the depth points.
    depths : np.ndarray
        Array of depth values.
    qualities_vector : QualitiesVector
        Struct containing vp, vs, and rho values.
    mesh_vector : MeshVector
        Struct containing mesh information such as latitude, longitude, and vs30.
    nz_tomography_data : TomographyData
        Struct containing New Zealand tomography data.
    partial_global_surface_depths : PartialGlobalSurfaceDepths
        Struct containing global surfaces depths.
    in_any_basin_lat_lon : bool
        Flag indicating if the point is in any basin latitude-longitude.
    on_boundary : bool
        Flag indicating if the point is on the boundary.
    interpolated_global_surface_values : dict
        Dictionary containing the interpolated values for vp, vs, and rho.
    logger : Logger, optional
        Logger instance for logging messages.

    """
    if logger is not None:
        logger.log(
            logging.DEBUG, f"Applying EP tomo (2010) model to {len(z_indices)} points"
        )

    # Convert surf_depth to meters (ascending order for searchsorted)
    surf_depth_ascending = (
        np.array(nz_tomography_data.surf_depth)[::-1] * 1000
    )  # Shape: (num_surfaces,)

    # Vectorized search for indices
    counts = len(surf_depth_ascending) - np.searchsorted(
        surf_depth_ascending, depths, side="right"
    )
    ind_above = counts - 1
    ind_below = counts

    # Ensure indices are within bounds
    ind_above = np.clip(ind_above, 0, len(nz_tomography_data.surfaces) - 1)
    ind_below = np.clip(ind_below, 0, len(nz_tomography_data.surfaces) - 1)

    # Group depths by (ind_above, ind_below) pairs to minimize interpolate calls
    unique_pairs = np.unique(np.stack((ind_above, ind_below), axis=1), axis=0)
    for idx_above, idx_below in unique_pairs:
        pair_mask = (ind_above == idx_above) & (ind_below == idx_below)
        z_indices_subset = z_indices[pair_mask]
        depths_subset = depths[pair_mask]

        # Interpolate vp, vs, rho simultaneously for this interval
        values = {}
        for vtype in VelocityTypes:
            val_above = interpolated_global_surface_values[vtype.name][idx_above]
            val_below = interpolated_global_surface_values[vtype.name][idx_below]

            dep_above = nz_tomography_data.surf_depth[idx_above] * 1000
            dep_below = nz_tomography_data.surf_depth[idx_below] * 1000
            val = linear_interpolation_vectorized(
                dep_above, dep_below, val_above, val_below, depths_subset
            )

            values[vtype.name] = val

        # Assign interpolated values
        qualities_vector.vp[z_indices_subset] = values["vp"]
        qualities_vector.vs[z_indices_subset] = values["vs"]
        qualities_vector.rho[z_indices_subset] = values["rho"]

    # Vectorized relative depth calculation
    relative_depths = partial_global_surface_depths.depths[1] - depths

    # Apply GTL and offshore smoothing
    if nz_tomography_data.gtl:
        if nz_tomography_data.special_offshore_tapering:
            # Determine if the offshore model should be applied (point-level condition)
            apply_offshore = (
                (mesh_vector.vs30 < 100)
                and (not in_any_basin_lat_lon)
                and (not on_boundary)
                and (mesh_vector.distance_from_shoreline > 0)
            )

            if apply_offshore:
                offshore_basinmodel_vectorized(
                    mesh_vector.distance_from_shoreline,
                    depths,
                    qualities_vector,
                    z_indices,
                    nz_tomography_data,
                )
            else:
                _apply_gtl(z_indices, relative_depths, qualities_vector, mesh_vector)
        else:
            _apply_gtl(z_indices, relative_depths, qualities_vector, mesh_vector)
