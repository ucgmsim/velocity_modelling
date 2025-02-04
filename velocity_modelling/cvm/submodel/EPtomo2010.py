import numpy as np
from velocity_modelling.cvm.registry import (
    QualitiesVector,
    MeshVector,
    TomographyData,
    PartialGlobalSurfaceDepths,
)
from velocity_modelling.cvm.interpolate import (
    linear_interpolation,
)

from Cant1D_v1 import main as v1d_sub_mod


def rho_from_vp_brocher(vp):
    """
    Calculate Rho given Vp from the Brocher correlation.

    Parameters
    ----------
    vp : float
        Primary wave velocity.

    Returns
    -------
    float
        Density (Rho) from the Brocher correlation.
    """
    density = vp * (
        1.6612 + vp * (-0.4721 + vp * (0.0671 + vp * (-0.0043 + 0.000106 * vp)))
    )
    return density


def vp_from_vs_brocher(vs):
    """
    Calculate Vp given vs from the Brocher correlation.

    Parameters
    ----------
    vs : float
        Secondary wave velocity.

    Returns
    -------
    float
        Primary wave velocity (Vp) from the Brocher correlation.
    """
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4
    return vp


def vs_from_vp_brocher(vp):
    """
    Calculate vs given Vp from the Brocher correlation.

    Parameters
    ----------
    vp : float
        Primary wave velocity.

    Returns
    -------
    float
        Secondary wave velocity (vs) from the Brocher correlation.
    """
    vs = 0.7858 - 1.2344 * vp + 0.7949 * vp**2 - 0.1238 * vp**3 + 0.0064 * vp**4
    return vs


def v30gtl(vs30, vt, z, zt, qualities_vector: QualitiesVector, zInd):
    """
    vs30 Geotechnical Layer (GTL) based on Ely (2010).

    Parameters
    ----------
    vs30 : float
        vs30 value.
    vt : float
        Velocity value.
    z : float
        Depth value.
    zt : float
        Taper depth.
    qualities_vector : QualitiesVector
        Struct containing the vp, vs, and rho values.
    zInd : int
        Index of the depth point.
    """
    a = 0.5
    b = 2.0 / 3.0
    c = 2

    z = z / zt  # z must be positive here
    f = z + b * (z - z * z)
    g = a - (a + 3.0 * c) * z + c * z * z + 2.0 * c * np.sqrt(z)
    qualities_vector.vs[zInd] = f * vt + g * vs30 / 1000  # vs30 must be in km/s
    qualities_vector.vp[zInd] = vp_from_vs_brocher(qualities_vector.vs[zInd])
    qualities_vector.rho[zInd] = rho_from_vp_brocher(qualities_vector.vp[zInd])


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
        v1d_sub_mod(zInd, dep, qualities_vector, velo_mod_1d_data)


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
    dep: float,
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

    count = 0
    # Find the index of the first "surface" above the data point in question
    while dep < nz_tomography_data.surf_depth[count] * 1000:
        count += 1
    ind_above = count - 1
    ind_below = count

    # Find the adjacent points for interpolation from the first surface
    adjacent_points = nz_tomography_data.surface[0][0].find_global_adjacent_points(
        mesh_vector
    )

    # Loop over the depth points and obtain the vp, vs, and rho values using interpolation between "surfaces"
    for i in range(3):
        surface_pointer_above = nz_tomography_data.surface[i][ind_above]
        surface_pointer_below = nz_tomography_data.surface[i][ind_below]

        val_above = surface_pointer_above.interpolate_global_surface(
            mesh_vector, adjacent_points
        )
        val_below = surface_pointer_below.interpolate_global_surface(
            mesh_vector, adjacent_points
        )

        dep_above = nz_tomography_data.surf_depth[ind_above] * 1000
        dep_below = nz_tomography_data.surf_depth[ind_below] * 1000
        val = linear_interpolation(dep_above, dep_below, val_above, val_below, dep)

        if i == 0:
            qualities_vector.vp[zInd] = val
        elif i == 1:
            qualities_vector.vs[zInd] = val
        elif i == 2:
            qualities_vector.rho[zInd] = val

    # Calculate relative depth
    relative_depth = partial_global_surface_depths.dep[1] - dep

    # Apply GTL and special offshore smoothing if necessary
    if gtl and not nz_tomography_data.special_offshore_tapering:
        if relative_depth <= 350:
            ely_taper_depth = 350
            v30gtl(
                mesh_vector.vs30,
                qualities_vector.vs[zInd],
                relative_depth,
                ely_taper_depth,
                qualities_vector,
                zInd,
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
                dep,
                qualities_vector,
                zInd,
                nz_tomography_data.offshore_basin_model_1d,
            )
        elif relative_depth <= 350:
            ely_taper_depth = 350
            v30gtl(
                mesh_vector.vs30,
                qualities_vector.vs[zInd],
                relative_depth,
                ely_taper_depth,
                qualities_vector,
                zInd,
            )
