import numpy as np

from velocity_modelling.cvm.geometry import MeshVector, AdjacentPoints
from velocity_modelling.cvm.velocity3d import QualitiesVector
from velocity_modelling.cvm.global_model import (
    PartialGlobalSurfaceDepths,
    interpolate_global_surface_numba,
    TomographyData,
)
from velocity_modelling.cvm.constants import VTYPE
from velocity_modelling.cvm.interpolate import (
    linear_interpolation,
    linear_interpolation_vectorized,
)
from velocity_modelling.cvm.gtl import v30gtl, v30gtl_vectorized
from velocity_modelling.cvm.submodel import Cant1D_v1


def off_shore_basin_model(
    shoreline_dist, dep, qualities_vector, zInd, velo_mod_1d_data
):
    """
    Offshore basin model.

    Parameters
    ----------
    shoreline_dist : float
        Distance from the shoreline.
    dep : float
        Depth value.
    qualities_vector : QualitiesVector
        Struct containing the vp, vs, and rho values.
    zInd : int
        Index of the depth point.
    velo_mod_1d_data : VelocityModel1D
        Struct containing 1D velocity model data.
    """
    offshore_depth = offshore_basin_depth(shoreline_dist)
    if offshore_depth < dep:
        Cant1D_v1.main(zInd, dep, qualities_vector, velo_mod_1d_data)


def offshore_basin_depth(shoreline_dist):
    """
    Calculate the offshore basin depth based on the distance from the shoreline.

    Parameters
    ----------
    shoreline_dist : float
        Distance from the shoreline.

    Returns
    -------
    float
        Basin depth.
    """
    if shoreline_dist > 50:
        basin_depth = -3000.0
    elif shoreline_dist > 20:
        basin_depth = -2000.0 - (((shoreline_dist - 20) / 30.0) * 1000.0)
    else:
        basin_depth = (shoreline_dist / 20.0) * -2000.0

    return basin_depth


def main(
    zInd: int,
    depth: float,
    qualities_vector: QualitiesVector,
    mesh_vector: MeshVector,
    nz_tomography_data: TomographyData,
    partial_global_surface_depths: PartialGlobalSurfaceDepths,
    gtl: bool,
    in_any_basin_lat_lon: bool,
    on_boundary: bool,
):
    """
    Calculate the rho, vp, and vs values at a single lat-long point for all the depths within this velocity submodel.

    Parameters
    ----------
    zInd : int
        Index of the depth point.
    dep : float
        Depth value.
    qualities_vector : QualitiesVector
        Struct containing the vp, vs, and rho values.
    mesh_vector : MeshVector
        Struct containing mesh information such as latitude, longitude, and vs30.
    nz_tomography_data : TomographyData
        Struct containing New Zealand tomography data.
    partial_global_surface_depths : PartialGlobalSurfaceDepths
        Struct containing global surfaces depths.
    gtl : bool
        Flag indicating whether GTL (Geotechnical Layer) is applied.
    in_any_basin_lat_lon : bool
        Flag indicating if the point is in any basin latitude-longitude.
    on_boundary : bool
        Flag indicating if the point is on the boundary.
    """

    # Convert surf_depth to a NumPy array for fast indexing
    surf_depth_ascending = (
        np.array(nz_tomography_data.surf_depth)[::-1] * 1000
    )  # convert to meters
    count = len(surf_depth_ascending) - np.searchsorted(
        surf_depth_ascending, depth, side="right"
    )

    # # Find the index of the first "surfaces" above the data point in question
    # while dep < nz_tomography_data.surf_depth[count] * 1000:
    #     count += 1

    # Indices for above and below
    ind_above, ind_below = count - 1, count

    # Vectorized search for adjacent points
    global_surf_read = nz_tomography_data.surfaces[0]["vp"]
    adjacent_points = AdjacentPoints.find_global_adjacent_points(
        global_surf_read.lati, global_surf_read.loni, mesh_vector.lat, mesh_vector.lon
    )

    # Loop over the depth points and obtain the vp, vs, and rho values using interpolation between "surfaces"
    for vtype in VTYPE:  # vp, vs, rho
        surface_pointer_above = nz_tomography_data.surfaces[ind_above][vtype.name]
        surface_pointer_below = nz_tomography_data.surfaces[ind_below][vtype.name]

        # val_above = interpolate_global_surface(
        #     surface_pointer_above, mesh_vector, adjacent_points
        # )
        # val_below = interpolate_global_surface(
        #     surface_pointer_below, mesh_vector, adjacent_points
        # )
        # Extract data once from mesh_vector and adjacent_points (reused across calls)
        lon = mesh_vector.lon
        lat = mesh_vector.lat
        lon_ind = adjacent_points.lon_ind  # np.ndarray
        lat_ind = adjacent_points.lat_ind  # np.ndarray
        in_surface_bounds = adjacent_points.in_surface_bounds
        in_lat_extension_zone = adjacent_points.in_lat_extension_zone
        in_lon_extension_zone = adjacent_points.in_lon_extension_zone
        in_corner_zone = adjacent_points.in_corner_zone
        lat_edge_ind = adjacent_points.lat_edge_ind
        lon_edge_ind = adjacent_points.lon_edge_ind
        corner_lon_ind = adjacent_points.corner_lon_ind
        corner_lat_ind = adjacent_points.corner_lat_ind

        # Replace the four calls with Numba-optimized version
        val_above = interpolate_global_surface_numba(
            surface_pointer_above.lati,
            surface_pointer_above.loni,
            surface_pointer_above.raster,
            lat,
            lon,
            lat_ind,
            lon_ind,
            in_surface_bounds,
            in_lat_extension_zone,
            in_lon_extension_zone,
            in_corner_zone,
            lat_edge_ind,
            lon_edge_ind,
            corner_lat_ind,
            corner_lon_ind,
        )

        val_below = interpolate_global_surface_numba(
            surface_pointer_below.lati,
            surface_pointer_below.loni,
            surface_pointer_below.raster,
            lat,
            lon,
            lat_ind,
            lon_ind,
            in_surface_bounds,
            in_lat_extension_zone,
            in_lon_extension_zone,
            in_corner_zone,
            lat_edge_ind,
            lon_edge_ind,
            corner_lat_ind,
            corner_lon_ind,
        )

        dep_above = nz_tomography_data.surf_depth[ind_above] * 1000
        dep_below = nz_tomography_data.surf_depth[ind_below] * 1000
        val = linear_interpolation(dep_above, dep_below, val_above, val_below, depth)

        if vtype.name == "vp":
            qualities_vector.vp[zInd] = val
        elif vtype.name == "vs":
            qualities_vector.vs[zInd] = val
        elif vtype.name == "rho":
            qualities_vector.rho[zInd] = val

    # Calculate relative depth
    # why depth[1]??
    relative_depth = (
        partial_global_surface_depths.depths[1] - depth
    )  # DEM minus the depth of the point

    # Apply GTL and special offshore smoothing if necessary
    if gtl and not nz_tomography_data.special_offshore_tapering:
        if relative_depth <= 350:
            (
                qualities_vector.vs[zInd],
                qualities_vector.vp[zInd],
                qualities_vector.rho[zInd],
            ) = v30gtl(
                mesh_vector.vs30,
                qualities_vector.vs[zInd],
                relative_depth,
                350,  # Ely (2010) GTL taper depth
            )
    elif gtl and nz_tomography_data.special_offshore_tapering:
        if (
            mesh_vector.vs30 < 100
            and not in_any_basin_lat_lon
            and not on_boundary
            and mesh_vector.distance_from_shoreline > 0
        ):
            off_shore_basin_model(
                mesh_vector.distance_from_shoreline,
                depth,
                qualities_vector,
                zInd,
                nz_tomography_data.offshore_basin_model_1d,
            )
        elif relative_depth <= 350:
            (
                qualities_vector.vs[zInd],
                qualities_vector.vp[zInd],
                qualities_vector.rho[zInd],
            ) = v30gtl(
                mesh_vector.vs30,
                qualities_vector.vs[zInd],
                relative_depth,
                350,  # Ely (2010) GTL taper depth
            )


def offshore_basin_depth_vectorized(shoreline_dist):
    """
    Calculate the offshore basin depth based on the distance from the shoreline.

    Parameters
    ----------
    shoreline_dist : float
        Distance from the shoreline.

    Returns
    -------
    float
        Basin depth.
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
    nz_tomography_data : TomographyData
        Struct containing New Zealand tomography data.
    """
    offshore_depths = offshore_basin_depth_vectorized(distance_from_shoreline)
    offshore_apply_mask = offshore_depths < depths
    z_indices_offshore = z_indices[offshore_apply_mask]
    depths_offshore = depths[offshore_apply_mask]
    if z_indices_offshore.size > 0:
        Cant1D_v1.main_vectorized(
            z_indices_offshore,
            depths_offshore,
            qualities_vector,
            nz_tomography_data.offshore_basin_model_1d,
        )


def main_vectorized(
    z_indices: np.ndarray,
    depths: np.ndarray,
    qualities_vector: QualitiesVector,
    mesh_vector: MeshVector,
    nz_tomography_data: TomographyData,
    partial_global_surface_depths: PartialGlobalSurfaceDepths,
    gtl: bool,
    in_any_basin_lat_lon: bool,
    on_boundary: bool,
):
    """
    Calculate rho, vp, and vs values for multiple lat-long-depth points within this velocity submodel.

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
    gtl : bool
        Flag indicating whether GTL (Geotechnical Layer) is applied.
    in_any_basin_lat_lon : bool
        Flag indicating if the point is in any basin latitude-longitude.
    on_boundary : bool
        Flag indicating if the point is on the boundary.
    """
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

    # Precompute adjacent points (same for all depths at this lat-lon)
    global_surf_read = nz_tomography_data.surfaces[0]["vp"]
    adjacent_points = AdjacentPoints.find_global_adjacent_points(
        global_surf_read.lati, global_surf_read.loni, mesh_vector.lat, mesh_vector.lon
    )

    # Extract reusable data from mesh_vector and adjacent_points
    lon = mesh_vector.lon
    lat = mesh_vector.lat
    lon_ind = adjacent_points.lon_ind
    lat_ind = adjacent_points.lat_ind
    in_surface_bounds = adjacent_points.in_surface_bounds
    in_lat_extension_zone = adjacent_points.in_lat_extension_zone
    in_lon_extension_zone = adjacent_points.in_lon_extension_zone
    in_corner_zone = adjacent_points.in_corner_zone
    lat_edge_ind = adjacent_points.lat_edge_ind
    lon_edge_ind = adjacent_points.lon_edge_ind
    corner_lon_ind = adjacent_points.corner_lon_ind
    corner_lat_ind = adjacent_points.corner_lat_ind

    # Group depths by (ind_above, ind_below) pairs to minimize interpolate calls
    unique_pairs = np.unique(np.stack((ind_above, ind_below), axis=1), axis=0)
    for idx_above, idx_below in unique_pairs:
        pair_mask = (ind_above == idx_above) & (ind_below == idx_below)
        z_indices_subset = z_indices[pair_mask]
        depths_subset = depths[pair_mask]

        # Interpolate vp, vs, rho simultaneously for this interval
        values = {}
        for vtype in VTYPE:
            surface_pointer_above = nz_tomography_data.surfaces[idx_above][vtype.name]
            surface_pointer_below = nz_tomography_data.surfaces[idx_below][vtype.name]

            # Interpolate for a single (lat, lon); scalar output
            val_above = interpolate_global_surface_numba(
                surface_pointer_above.lati,
                surface_pointer_above.loni,
                surface_pointer_above.raster,
                lat,
                lon,
                lat_ind,
                lon_ind,
                in_surface_bounds,
                in_lat_extension_zone,
                in_lon_extension_zone,
                in_corner_zone,
                lat_edge_ind,
                lon_edge_ind,
                corner_lat_ind,
                corner_lon_ind,
            )
            val_below = interpolate_global_surface_numba(
                surface_pointer_below.lati,
                surface_pointer_below.loni,
                surface_pointer_below.raster,
                lat,
                lon,
                lat_ind,
                lon_ind,
                in_surface_bounds,
                in_lat_extension_zone,
                in_lon_extension_zone,
                in_corner_zone,
                lat_edge_ind,
                lon_edge_ind,
                corner_lat_ind,
                corner_lon_ind,
            )

            # Vectorized linear interpolation across depths
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
    if gtl and not nz_tomography_data.special_offshore_tapering:
        gtl_mask = relative_depths <= 350
        if np.any(gtl_mask):
            z_indices_gtl = z_indices[gtl_mask]
            vs_gtl = qualities_vector.vs[z_indices_gtl]
            relative_depths_gtl = relative_depths[gtl_mask]
            vs_new, vp_new, rho_new = v30gtl_vectorized(
                mesh_vector.vs30, vs_gtl, relative_depths_gtl, 350
            )
            qualities_vector.vs[z_indices_gtl] = vs_new
            qualities_vector.vp[z_indices_gtl] = vp_new
            qualities_vector.rho[z_indices_gtl] = rho_new
    elif gtl and nz_tomography_data.special_offshore_tapering:
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
            # Apply GTL only if apply_offshore is False
            gtl_mask = relative_depths <= 350
            if np.any(gtl_mask):
                z_indices_gtl = z_indices[gtl_mask]
                vs_gtl = qualities_vector.vs[z_indices_gtl]
                relative_depths_gtl = relative_depths[gtl_mask]
                vs_new, vp_new, rho_new = v30gtl_vectorized(
                    mesh_vector.vs30, vs_gtl, relative_depths_gtl, 350
                )
                qualities_vector.vs[z_indices_gtl] = vs_new
                qualities_vector.vp[z_indices_gtl] = vp_new
                qualities_vector.rho[z_indices_gtl] = rho_new
