import numpy as np

from registry import (
    BasinData,
    InBasin,
    PartialBasinSurfaceDepths,
    GlobalSurfaceRead,
    AdjacentPoints,
    PartialGlobalSurfaceDepths,
    GlobalSurfaces,
    MeshVector,
)  # , CalculationLog


def interpolate_global_surface_depths(
    partial_global_surface_depth: PartialGlobalSurfaceDepths,
    global_surfaces: GlobalSurfaces,
    mesh_vector: MeshVector,
    calculation_log,
):
    """
    Interpolate the surface depths at the lat lon location given in mesh_vector.

    Parameters
    ----------
    global_surfaces : GlobalSurfaces
        Object containing pointers to global surfaces.
    mesh_vector : MeshVector
        Object containing a single lat lon point with one or more depths.
    calculation_log : CalculationLog
        Object containing calculation data and output directory.
    """
    for i in range(global_surfaces.nSurf):
        global_surf_read = global_surfaces.surf[i]
        adjacent_points = AdjacentPoints()
        adjacent_points = global_surf_read.find_global_adjacent_points(mesh_vector)
        partial_global_surface_depth.dep[i] = (
            global_surf_read.interpolate_global_surface(mesh_vector, adjacent_points)
        )

    partial_global_surface_depth.nSurfDep = global_surfaces.nSurf
    for i in range(global_surfaces.nSurf - 1, 0, -1):
        top_val = partial_global_surface_depth.dep[i - 1]
        bot_val = partial_global_surface_depth.dep[i]
        if top_val < bot_val:
            partial_global_surface_depth.dep[i] = top_val
            # calculation_log.nPointsGlobalSurfacesEnforced += 1


def bi_linear_interpolation(
    x1: np.float64,
    x2: np.float64,
    y1: np.float64,
    y2: np.float64,
    q11: np.float64,
    q12: np.float64,
    q21: np.float64,
    q22: np.float64,
    x: np.float64,
    y: np.float64,
):
    """
    Perform bilinear interpolation between four points.

    Parameters
    ----------
    x1, x2, y1, y2 : float
        Coordinates of the points.
    q11, q12, q21, q22 : float
        Values at the four points.
    x, y : float
        Coordinates of the point to interpolate.

    Returns
    -------
    float
        Interpolated value at the given x, y.
    """
    A = q11 * (x2 - x) * (y2 - y)
    B = q21 * (x - x1) * (y2 - y)
    C = q12 * (x2 - x) * (y - y1)
    D = q22 * (x - x1) * (y - y1)
    E = 1 / (x2 - x1) / (y2 - y1)

    return (A + B + C + D) * E


def interpolate_global_surface(
    global_surface_read: GlobalSurfaceRead, mesh_vector: MeshVector, adjacent_points
):
    """
    Interpolate the global surface value at a given latitude and longitude.

    Parameters:
    lat (float): Latitude of the point for interpolation.
    lon (float): Longitude of the point for interpolation.
    adjacent_points (AdjacentPoints): Object containing indices of points adjacent to the lat-lon for interpolation.

    Returns:
    float: Interpolated value at the given lat-lon.
    """

    x1 = global_surface_read.loni[adjacent_points.lon_ind[0]]
    x2 = global_surface_read.loni[adjacent_points.lon_ind[1]]

    y1 = global_surface_read.lati[adjacent_points.lat_ind[0]]
    y2 = global_surface_read.lati[adjacent_points.lat_ind[1]]

    q11 = global_surface_read.raster[adjacent_points.lon_ind[0]][
        adjacent_points.lat_ind[0]
    ]
    q12 = global_surface_read.raster[adjacent_points.lon_ind[0]][
        adjacent_points.lat_ind[1]
    ]
    q21 = global_surface_read.raster[adjacent_points.lon_ind[1]][
        adjacent_points.lat_ind[0]
    ]
    q22 = global_surface_read.raster[adjacent_points.lon_ind[1]][
        adjacent_points.lat_ind[1]
    ]

    return bi_linear_interpolation(
        x1, x2, y1, y2, q11, q12, q21, q22, mesh_vector.Lon, mesh_vector.Lat
    )


def interpolate_basin_surface_depths(
    basin_data: BasinData,
    in_basin: InBasin,
    partial_basin_surface_depths: PartialBasinSurfaceDepths,
    mesh_vector: MeshVector,
):
    """
    Determine if a lat-lon point is in a basin, if so interpolate the basin surface depths, enforce their hierarchy, then determine which depth points lie within the basin limits.

    Parameters
    ----------
    in_basin : InBasin
        Struct containing flags to indicate if lat-lon point - depths lie within the basin.
    partial_basin_surface_depths : PartialBasinSurfaceDepths
        Struct containing depths for all applicable basin surfaces at one lat-lon location.
    mesh_vector : MeshVector
        Struct containing a single lat-lon point with one or more depths.
    """
    basin_data.determine_if_within_basin_lat_lon(mesh_vector, in_basin)
    basin_data.determine_basin_surface_depths(
        in_basin, partial_basin_surface_depths, mesh_vector
    )
    basin_data.enforce_basin_surface_depths(partial_basin_surface_depths)
