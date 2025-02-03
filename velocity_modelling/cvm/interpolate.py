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
    lat = mesh_vector.Lat
    lon = mesh_vector.Lon

    lat1 = global_surface_read.lati[adjacent_points.lat_ind[0]]
    lat2 = global_surface_read.lati[adjacent_points.lat_ind[1]]
    lon1 = global_surface_read.loni[adjacent_points.lon_ind[0]]
    lon2 = global_surface_read.loni[adjacent_points.lon_ind[1]]

    f11 = global_surface_read.raster[adjacent_points.lat_ind[0]][
        adjacent_points.lon_ind[0]
    ]
    f12 = global_surface_read.raster[adjacent_points.lat_ind[0]][
        adjacent_points.lon_ind[1]
    ]
    f21 = global_surface_read.raster[adjacent_points.lat_ind[1]][
        adjacent_points.lon_ind[0]
    ]
    f22 = global_surface_read.raster[adjacent_points.lat_ind[1]][
        adjacent_points.lon_ind[1]
    ]

    # bilinear interpolation between the four points
    interpolated_value = (
        f11 * (lat2 - lat) * (lon2 - lon)
        + f21 * (lat - lat1) * (lon2 - lon)
        + f12 * (lat2 - lat) * (lon - lon1)
        + f22 * (lat - lat1) * (lon - lon1)
    ) / ((lat2 - lat1) * (lon2 - lon1))

    return interpolated_value


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
        partial_basin_surface_depths, mesh_vector.Lat, mesh_vector.Lon
    )
    basin_data.enforce_basin_surface_depths(partial_basin_surface_depths, mesh_vector)
