import numpy as np

from velocity_modelling.cvm.geometry import AdjacentPoints
from velocity_modelling.cvm.registry import TomographyData
from velocity_modelling.cvm.velocity import MeshVector, QualitiesVector
from velocity_modelling.cvm.global_model import (
    PartialGlobalSurfaceDepths,
    interpolate_global_surface,
)
from velocity_modelling.cvm.constants import VTYPE
from velocity_modelling.cvm.interpolate import linear_interpolation
from velocity_modelling.cvm.gtl import v30gtl
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
    velo_mod_1d_data : VeloMod1DData
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
        Struct containing global surface depths.
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

    # # Find the index of the first "surface" above the data point in question
    # while dep < nz_tomography_data.surf_depth[count] * 1000:
    #     count += 1

    # Indices for above and below
    ind_above, ind_below = count - 1, count

    # Vectorized search for adjacent points
    global_surf_read = nz_tomography_data.surface[0]["vp"]
    adjacent_points = AdjacentPoints.find_global_adjacent_points(
        global_surf_read.lati, global_surf_read.loni, mesh_vector.lat, mesh_vector.lon
    )

    # Loop over the depth points and obtain the vp, vs, and rho values using interpolation between "surfaces"
    for vtype in VTYPE:  # vp, vs, rho
        surface_pointer_above = nz_tomography_data.surface[ind_above][vtype.name]
        surface_pointer_below = nz_tomography_data.surface[ind_below][vtype.name]

        val_above = interpolate_global_surface(
            surface_pointer_above, mesh_vector, adjacent_points
        )
        val_below = interpolate_global_surface(
            surface_pointer_below, mesh_vector, adjacent_points
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
        partial_global_surface_depths.depth[1] - depth
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
