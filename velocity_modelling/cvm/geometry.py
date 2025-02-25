from logging import Logger
from typing import Any
from typing import List, Dict

import numpy as np
from numba import njit
from velocity_modelling.cvm.constants import (
    MAX_LAT_SURFACE_EXTENSION,
    MAX_LON_SURFACE_EXTENSION,
    EARTH_RADIUS_MEAN,
    ERAD,
    RPERD,
    LON_GRID_DIM_MAX,
    LAT_GRID_DIM_MAX,
    DEP_GRID_DIM_MAX,
)


class GlobalMesh:
    def __init__(self, nx: int, ny: int, nz: int):
        """
        Initialize the GlobalMesh.

        Parameters
        ----------
        nx : int
            The number of points in the X direction.
        ny : int
            The number of points in the Y direction.
        nz : int
            The number of points in the Z direction.
        """
        self.lon = np.zeros((nx, ny))
        self.lat = np.zeros((nx, ny))
        self.max_lat = 0.0
        self.min_lat = 0.0
        self.max_lon = 0.0
        self.min_lon = 0.0
        self.x = np.zeros(nx)
        self.y = np.zeros(ny)
        self.z = np.zeros(nz)

    @property
    def nx(self):
        return len(self.x)

    @property
    def ny(self):
        return len(self.y)

    @property
    def nz(self):
        return len(self.z)


class PartialGlobalMesh:
    def __init__(self, global_mesh: GlobalMesh, lat_ind: int):
        """
        Initialize the PartialGlobalMesh.

        Parameters
        ----------
        nx : int
            The number of points in the X direction.
        nz : int
            The number of points in the Z direction.
        """

        self.x = global_mesh.x.copy()
        self.y = global_mesh.y[lat_ind]
        self.z = global_mesh.z.copy()
        self.lat = global_mesh.lat[:, lat_ind].copy()
        self.lon = global_mesh.lon[:, lat_ind].copy()

    @property
    def nx(self):
        return len(self.x)

    @property
    def nz(self):
        return len(self.z)


class MeshVector:
    def __init__(self, partial_global_mesh: PartialGlobalMesh, lon_ind: int):

        self.lat = partial_global_mesh.lat[lon_ind]
        self.lon = partial_global_mesh.lon[lon_ind]
        self.z = partial_global_mesh.z.copy()
        self.vs30 = None
        self.distance_from_shoreline = None

    @property
    def nz(self):
        return len(self.z)


class ModelExtent:
    def __init__(self, vm_params: Dict):
        """
        Initialize the ModelExtent.

        Parameters
        ----------
        vm_params : Dict
            The velocity model parameters.
        """
        self.origin_lat = vm_params["MODEL_LAT"]
        self.origin_lon = vm_params["MODEL_LON"]
        self.origin_rot = vm_params["MODEL_ROT"]  # in degrees
        self.xmax = vm_params["extent_x"]
        self.ymax = vm_params["extent_y"]
        self.zmax = vm_params["extent_zmax"]
        self.zmin = vm_params["extent_zmin"]
        self.h_depth = vm_params["hh"]
        self.h_lat_lon = vm_params["hh"]
        self.nx = vm_params["nx"]
        self.ny = vm_params["ny"]
        self.nz = vm_params["nz"]


# def point_on_vertex(basin_data: BasinData, boundary_ind: int, mesh_vector: MeshVector) -> bool:


@njit
def point_on_vertex(
    boundary_lats: np.ndarray, boundary_lons: np.ndarray, lat: float, lon: float
) -> bool:
    """
    Check if a point lies on a vertex of a basin boundary.

    Parameters
    ----------
    boundary_lats : np.ndarray
        Array of latitudes of the boundary vertices.
    boundary_lons : np.ndarray
        Array of longitudes of the boundary vertices.
    lat : float
        Latitude of the point to check.
    lon : float
        Longitude of the point to check.

    Returns
    -------
    bool
        True if the point lies on a vertex of the boundary, False otherwise.
    """
    # Vectorized comparison using NumPy with tolerance
    # lat_matches = np.isclose(boundary_lats, lat, rtol=1e-07, atol=1e-07)
    # lon_matches = np.isclose(boundary_lons, lon, rtol=1e-07, atol=1e-07)
    lat_matches = (
        np.abs(boundary_lats - lat) <= 1e-07
    )  # np.isclose is not supported by njit
    lon_matches = np.abs(boundary_lons - lon) <= 1e-07

    matches = np.logical_and(lat_matches, lon_matches)

    return np.any(matches)


@njit
def find_edge_inds(
    lati: np.ndarray,
    loni: np.ndarray,
    edge_type: int,
    max_lat: float,
    min_lat: float,
    max_lon: float,
    min_lon: float,
):
    nlat = len(lati)
    nlon = len(loni)
    lat_edge_ind = 0
    lon_edge_ind = 0

    if edge_type == 1:  # north edge
        if max_lat == lati[0]:
            lat_edge_ind = 0
        elif max_lat == lati[-1]:
            lat_edge_ind = nlat - 1
        else:
            raise ValueError("Point lies outside of surface bounds.")
    elif edge_type == 3:  # south edge
        if min_lat == lati[0]:
            lat_edge_ind = 0
        elif min_lat == lati[-1]:
            lat_edge_ind = nlat - 1
        else:
            raise ValueError("Point lies outside of surface bounds.")
    elif edge_type == 2:  # east edge
        if max_lon == loni[0]:
            lon_edge_ind = 0
        elif max_lon == loni[-1]:
            lon_edge_ind = nlon - 1
        else:
            raise ValueError("Point lies outside of surface bounds.")
    elif edge_type == 4:  # west edge
        if min_lon == loni[0]:
            lon_edge_ind = 0
        elif min_lon == loni[-1]:
            lon_edge_ind = nlon - 1
        else:
            raise ValueError("Point lies outside of surface bounds.")
    else:
        raise ValueError("Point lies outside of surface bounds.")

    return lat_edge_ind, lon_edge_ind


@njit
def find_corner_inds(lati: np.ndarray, loni: np.ndarray, lat: float, lon: float):
    nlat = len(lati)
    nlon = len(loni)
    corner_lat_ind = 0
    corner_lon_ind = 0

    if np.abs(lat - lati[0]) <= 1e-07:
        corner_lat_ind = 0
    elif np.abs(lat - lati[-1]) <= 1e-07:
        corner_lat_ind = nlat - 1
    else:
        raise ValueError(
            f"Point lies outside of surface bounds. Lat {lat} outside [{lati[0]} {lati[-1]}]"
        )

    if np.abs(lon - loni[0]) <= 1e-07:
        corner_lon_ind = 0
    elif np.abs(lon - loni[-1]) <= 1e-07:
        corner_lon_ind = nlon - 1
    else:
        raise ValueError(
            f"Point lies outside of surface bounds. Lon {lon} outside [{loni[0]} {loni[-1]}]"
        )

    return corner_lat_ind, corner_lon_ind


class AdjacentPoints:
    def __init__(self):
        self.in_surface_bounds = False
        self.lat_ind = [0, 0]
        self.lon_ind = [0, 0]
        self.in_lat_extension_zone = False
        self.lat_extension_type = 0
        self.lon_edge_ind = 0
        self.in_lon_extension_zone = False
        self.lon_extension_type = 0
        self.lat_edge_ind = 0
        self.in_corner_zone = False
        self.corner_lat_ind = 0
        self.corner_lon_ind = 0

    @classmethod
    def find_basin_adjacent_points(
        cls, lati: np.ndarray, loni: np.ndarray, lat: float, lon: float
    ):
        """
        Find the adjacent points to the given latitude and longitude in the basin surface.

        Parameters
        ----------
        lati : np.ndarray
            Array of latitudes of the basin surface.
        loni : np.ndarray
            Array of longitudes of the basin surface.
        lat : float
            Latitude of the point to check.
        lon : float
            Longitude of the point to check.

        Returns
        -------
        None
        """
        adjacent_points = cls()

        # Handle latitude
        if lati[0] > lati[-1]:  # descending order
            lat_idx = np.searchsorted(lati[::-1], lat)
            lat_idx = len(lati) - lat_idx - 1
        else:  # ascending order
            lat_idx = np.searchsorted(lati, lat)

        if 0 < lat_idx < len(lati):
            adjacent_points.lat_ind = [lat_idx - 1, lat_idx]
        elif lat_idx == 0 and lati[0] == lat:
            adjacent_points.lat_ind = [0, 1]
        elif lat_idx == len(lati) and lati[-1] == lat:
            adjacent_points.lat_ind = [len(lati) - 1, len(lati)]

        # Handle longitude
        if loni[0] > loni[-1]:  # descending order
            lon_idx = np.searchsorted(loni[::-1], lon)
            lon_idx = len(loni) - lon_idx - 1
        else:  # ascending order
            lon_idx = np.searchsorted(loni, lon)

        if 0 < lon_idx < len(loni):
            adjacent_points.lon_ind = [lon_idx - 1, lon_idx]
        elif lon_idx == 0 and loni[0] == lon:
            adjacent_points.lon_ind = [0, 1]
        elif lon_idx == len(loni) and loni[-1] == lon:
            adjacent_points.lon_ind = [len(loni) - 1, len(loni)]

        if adjacent_points.lat_ind != [0, 0] and adjacent_points.lon_ind != [0, 0]:
            adjacent_points.in_surface_bounds = True
        else:
            print(
                f"Error, basin point lies outside of the extent of the basin surface ({lon}, {lat})."
            )
        return adjacent_points

    @classmethod
    def find_global_adjacent_points(
        cls, lati: np.ndarray, loni: np.ndarray, lat: float, lon: float
    ):
        """
        Find the adjacent points to the given latitude and longitude in the global surface.

        Parameters
        ----------
        lati : np.ndarray
            Array of latitudes of the global surface.
        loni : np.ndarray
            Array of longitudes of the global surface.
        lat : float
            Latitude of the point to check.
        lon : float
            Longitude of the point to check.

        Returns
        -------
        None
        """
        max_lat = np.max(lati)
        min_lat = np.min(lati)
        max_lon = np.max(loni)
        min_lon = np.min(loni)
        nlat = len(lati)
        nlon = len(loni)

        adjacent_points = cls()

        lat_assigned_flag = False
        lon_assigned_flag = False

        if lati[0] < lat <= lati[-1] or lati[0] > lat >= lati[-1]:
            is_ascending = lati[0] < lati[-1]
            lati = (
                lati if is_ascending else lati[::-1]
            )  # reverse the array if sorted in descending order

            index = np.searchsorted(lati, lat)
            index = (
                nlat - index if not is_ascending else index
            )  # reverse the index if sorted in descending order

            if 0 < index < nlat:
                adjacent_points.lat_ind[0] = index - 1
                adjacent_points.lat_ind[1] = index
                lat_assigned_flag = True

        if loni[0] < lon <= loni[-1] or loni[0] > lon >= loni[-1]:
            is_ascending = loni[0] < loni[-1]
            loni = (
                loni if is_ascending else loni[::-1]
            )  # reverse the array if sorted in descending order

            index = np.searchsorted(loni, lon)
            index = (
                nlon - index if not is_ascending else index
            )  # reverse the index if sorted in descending order

            if 0 < index < nlon:
                adjacent_points.lon_ind[0] = index - 1
                adjacent_points.lon_ind[1] = index
                lon_assigned_flag = True

        if lat_assigned_flag and lon_assigned_flag:
            adjacent_points.in_surface_bounds = True
        else:
            if lon_assigned_flag and not lat_assigned_flag:
                if (lat - max_lat) <= MAX_LAT_SURFACE_EXTENSION and lat >= max_lat:
                    adjacent_points.in_lat_extension_zone = True
                    adjacent_points.lat_edge_ind, adjacent_points.lon_edge_ind = (
                        find_edge_inds(
                            lati, loni, 1, max_lat, min_lat, max_lon, min_lon
                        )
                    )

                elif (min_lat - lat) <= MAX_LAT_SURFACE_EXTENSION and lat <= min_lat:
                    adjacent_points.in_lat_extension_zone = True
                    adjacent_points.lat_edge_ind, adjacent_points.lon_edge_ind = (
                        find_edge_inds(
                            lati, loni, 3, max_lat, min_lat, max_lon, min_lon
                        )
                    )

            if lat_assigned_flag and not lon_assigned_flag:
                if (min_lon - lon) <= MAX_LON_SURFACE_EXTENSION and lon <= min_lon:
                    adjacent_points.in_lon_extension_zone = True
                    adjacent_points.lat_edge_ind, adjacent_points.lon_edge_ind = (
                        find_edge_inds(
                            lati, loni, 4, max_lat, min_lat, max_lon, min_lon
                        )
                    )

                elif (lon - max_lon) <= MAX_LON_SURFACE_EXTENSION and lon >= max_lon:
                    adjacent_points.in_lon_extension_zone = True
                    adjacent_points.lat_edge_ind, adjacent_points.lon_edge_ind = (
                        find_edge_inds(
                            lati, loni, 2, max_lat, min_lat, max_lon, min_lon
                        )
                    )

            # four cases for corner zones
            if (
                (lat - max_lat) <= MAX_LAT_SURFACE_EXTENSION
                and (min_lon - lon) <= MAX_LON_SURFACE_EXTENSION
                and lon <= min_lon
                and lat >= max_lat
            ):
                adjacent_points.corner_lat_ind, adjacent_points.corner_lon_ind = (
                    find_corner_inds(lati, loni, max_lat, min_lon)
                )
            elif (
                lat - max_lat <= MAX_LAT_SURFACE_EXTENSION
                and lon - max_lon <= MAX_LON_SURFACE_EXTENSION
                and lon >= max_lon
                and lat >= max_lat
            ):
                adjacent_points.corner_lat_ind, adjacent_points.corner_lon_ind = (
                    find_corner_inds(lati, loni, max_lat, max_lon)
                )

            elif (
                min_lat - lat <= MAX_LAT_SURFACE_EXTENSION
                and min_lon - lon <= MAX_LON_SURFACE_EXTENSION
                and lon <= min_lon
                and lat <= min_lat
            ):
                adjacent_points.corner_lat_ind, adjacent_points.corner_lon_ind = (
                    find_corner_inds(lati, loni, min_lat, min_lon)
                )
            elif (
                min_lat - lat <= MAX_LAT_SURFACE_EXTENSION
                and lon - max_lon <= MAX_LON_SURFACE_EXTENSION
                and lon >= max_lon
                and lat <= min_lat
            ):
                adjacent_points.corner_lat_ind, adjacent_points.corner_lon_ind = (
                    find_corner_inds(lati, loni, min_lat, max_lon)
                )

            adjacent_points.in_corner_zone = (
                True  # TODO: this is always True, so why have it?
            )

            if not (
                adjacent_points.in_lat_extension_zone
                or adjacent_points.in_lon_extension_zone
                or adjacent_points.in_corner_zone
            ):
                raise ValueError(
                    f"Point does not lie in any global surface extension. {lon} {lat}"
                )

        return adjacent_points


class SmoothingBoundary:
    def __init__(self, xpts, ypts):
        """
        Initialize the SmoothingBoundary.
        """
        self.x = xpts
        self.y = ypts
        self.n = len(xpts)
        assert len(self.x) == len(self.y)

    def determine_if_lat_lon_within_smoothing_region(self, mesh_vector: MeshVector):
        """
        Determine the closest index within the smoothing boundary to the given mesh vector coordinates.

        Parameters:
        mesh_vector (MeshVector): Object containing mesh vector data.

        Returns:
        int: Index of the closest point in the smoothing boundary.
        """
        closest_ind, distance = self.brute_force(mesh_vector)

        return closest_ind, distance / 1000  # return distance in km

    def brute_force(self, mesh_vector: MeshVector):
        """
        Determine the closest index within the smoothing boundary to the given mesh vector coordinates.

        Parameters:
        mesh_vector (MeshVector): Object containing mesh vector data.

        Returns:
        int: Index of the closest point in the smoothing boundary.
        """
        boundary_points = np.column_stack((self.y, self.x))
        # the below wasn't giving the exact same results as the original C code
        # distances = coordinates.distance_between_wgs_depth_coordinates(
        #     boundary_points, np.array([mesh_vector.lat, mesh_vector.lon])
        # )
        distances = (
            lat_lon_to_distance(boundary_points, mesh_vector.lat, mesh_vector.lon)
            * 1000
        )  # convert to meters
        closest_ind = np.argmin(distances)
        return closest_ind, distances[closest_ind]


@njit
def lat_lon_to_distance(
    lat_lon_array: np.ndarray, origin_lat: float, origin_lon: float
) -> np.ndarray:
    """
    Calculate the distance from an origin point to a set of latitude and longitude points.

    Parameters
    ----------
    lat_lon_array : np.ndarray
        An array of shape (N, 2) where N is the number of points. Each row contains [latitude, longitude].
    origin_lat : float
        The latitude of the origin point.
    origin_lon : float
        The longitude of the origin point.

    Returns
    -------
    np.ndarray
        An array of distances from the origin point to each point in lat_lon_array.
    """
    ref_lon = np.deg2rad(origin_lon)
    ref_lat = np.deg2rad(origin_lat)

    lat_rad = np.deg2rad(lat_lon_array[:, 0])
    lon_rad = np.deg2rad(lat_lon_array[:, 1])

    dLon = lon_rad - ref_lon

    dz = np.sin(lat_rad) - np.sin(ref_lat)
    dx = np.cos(dLon) * np.cos(lat_rad) - np.cos(ref_lat)
    dy = np.sin(dLon) * np.cos(lat_rad)

    distances = (
        np.arcsin(np.sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * EARTH_RADIUS_MEAN
    )

    return distances


@njit
def great_circle_projection(
    x: np.ndarray,
    y: np.ndarray,
    amat: np.ndarray,
    erad: float = ERAD,
    g0: float = 0,
    b0: float = 0,
) -> tuple[np.ndarray, Any]:
    """
    Project x, y coordinates to geographic coordinates (longitude, latitude) using a great circle projection.

    Parameters
    ----------
    x : np.ndarray
        X-coordinates.
    y : np.ndarray
        Y-coordinates.
    amat : np.ndarray
        Transformation matrix.
    erad : float, optional
        Earth's radius (default is ERAD).
    g0 : float, optional
        Initial longitude (default is 0).
    b0 : float, optional
        Initial latitude (default is 0).

    Returns
    -------
    tuple[np.ndarray, Any]
        Computed latitude and longitude arrays.
    """
    cosB = np.cos(x / erad - b0)
    sinB = np.sin(x / erad - b0)

    cosG = np.cos(y / erad - g0)
    sinG = np.sin(y / erad - g0)

    xp = sinG * cosB * np.sqrt(1 + sinB * sinB * sinG * sinG)
    yp = sinB * cosG * np.sqrt(1 + sinB * sinB * sinG * sinG)
    zp = np.sqrt(1 - xp * xp - yp * yp)
    coords = np.stack((xp, yp, zp), axis=0)

    #    xg, yg, zg = np.tensordot(amat, coords, axes=([1], [0]))
    xg = amat[0, 0] * coords[0] + amat[0, 1] * coords[1] + amat[0, 2] * coords[2]
    yg = amat[1, 0] * coords[0] + amat[1, 1] * coords[1] + amat[1, 2] * coords[2]
    zg = amat[2, 0] * coords[0] + amat[2, 1] * coords[1] + amat[2, 2] * coords[2]

    lat = np.where(
        np.isclose(zg, 0),
        0,
        90 - np.arctan(np.sqrt(xg**2 + yg**2) / zg) / RPERD - np.where(zg < 0, 180, 0),
    )

    lon = np.where(
        np.isclose(xg, 0), 0, np.arctan(yg / xg) / RPERD - np.where(xg < 0, 180, 0)
    )
    lon = lon % 360
    return lat, lon


def gen_full_model_grid_great_circle(
    model_extent: ModelExtent, logger: Logger
) -> GlobalMesh:
    """
    Generate the grid of latitude, longitude, and depth points using the point radial distance method.

    Parameters
    ----------
    model_extent : ModelExtent
        Object containing the extent, spacing, and version of the model.
    logger : Logger
        Logger for logging information.

    Returns
    -------
    GlobalMesh
        Object containing the generated grid of latitude, longitude, and depth points.
    """
    nx = int(np.round(model_extent.xmax / model_extent.h_lat_lon))
    ny = int(np.round(model_extent.ymax / model_extent.h_lat_lon))
    nz = int(np.round((model_extent.zmax - model_extent.zmin) / model_extent.h_depth))

    global_mesh = GlobalMesh(nx, ny, nz)

    global_mesh.max_lat = -180
    global_mesh.min_lat = 0
    global_mesh.max_lon = 0
    global_mesh.min_lon = 180

    assert nx == model_extent.nx
    assert ny == model_extent.ny
    assert nz == model_extent.nz

    if any(
        [
            nx >= LON_GRID_DIM_MAX,
            ny >= LAT_GRID_DIM_MAX,
            nz >= DEP_GRID_DIM_MAX,
        ]
    ):
        raise ValueError(
            f"Grid dimensions exceed maximum allowable values. X={LON_GRID_DIM_MAX}, Y={LAT_GRID_DIM_MAX}, Z={DEP_GRID_DIM_MAX}"
        )

    if nz != 1:
        logger.info(f"Number of model points. nx: {nx}, ny: {ny}, nz: {nz}.")

    # for i in range(nx):
    #     global_mesh.x[i] = (
    #         0.5 * model_extent.h_lat_lon
    #         + model_extent.h_lat_lon * i
    #         - 0.5 * model_extent.xmax
    #     )
    global_mesh.x = (
        0.5 * model_extent.h_lat_lon
        + model_extent.h_lat_lon * np.arange(nx)
        - 0.5 * model_extent.xmax
    )

    # for i in range(ny):
    #     global_mesh.y[i] = (
    #         0.5 * model_extent.h_lat_lon
    #         + model_extent.h_lat_lon * i
    #         - 0.5 * model_extent.ymax
    #     )
    global_mesh.y = (
        0.5 * model_extent.h_lat_lon
        + model_extent.h_lat_lon * np.arange(ny)
        - 0.5 * model_extent.ymax
    )

    # for i in range(nz):
    #     global_mesh.z[i] = -1000 * (
    #         model_extent.zmin + model_extent.h_depth * (i + 0.5)
    #     )
    global_mesh.z = -1000 * (
        0.5 * model_extent.h_depth
        + model_extent.h_depth * np.arange(nz)
        + model_extent.zmin
    )

    arg = model_extent.origin_rot * RPERD
    cosA = np.cos(arg)
    sinA = np.sin(arg)

    arg = (90.0 - model_extent.origin_lat) * RPERD
    cosT = np.cos(arg)
    sinT = np.sin(arg)

    arg = model_extent.origin_lon * RPERD
    cosP = np.cos(arg)
    sinP = np.sin(arg)

    amat = np.array(
        [
            cosA * cosT * cosP + sinA * sinP,
            sinA * cosT * cosP - cosA * sinP,
            sinT * cosP,
            cosA * cosT * sinP - sinA * cosP,
            sinA * cosT * sinP + cosA * cosP,
            sinT * sinP,
            -cosA * sinT,
            -sinA * sinT,
            cosT,
        ]
    ).reshape((3, 3))

    det = np.linalg.det(amat)
    ainv = np.linalg.inv(amat) / det

    g0 = 0.0
    b0 = 0.0

    x, y = np.meshgrid(global_mesh.x[:nx], global_mesh.y[:ny], indexing="ij")
    lat_lon = great_circle_projection(x, y, amat, ERAD, g0, b0)
    (
        global_mesh.lat[:nx, :ny],
        global_mesh.lon[:nx, :ny],
    ) = lat_lon

    global_mesh.max_lat = np.max(global_mesh.lat)
    global_mesh.max_lon = np.max(global_mesh.lon)
    global_mesh.min_lat = np.min(global_mesh.lat)
    global_mesh.min_lon = np.min(global_mesh.lon)

    logger.info("Completed Generation of Model Grid.")
    return global_mesh


def extract_partial_mesh(global_mesh: GlobalMesh, lat_ind: int) -> PartialGlobalMesh:
    """
    Extract one slice of values from the global mesh, i.e., nx x ny x nz becomes nx x 1 x nz.

    Parameters
    ----------
    global_mesh : GlobalMesh
        The global mesh containing the full model grid (lat, lon, and depth points).
    lat_ind : int
        The y index of the slice of the global grid to be extracted.

    Returns
    -------
    PartialGlobalMesh
        A struct containing a slice of the global mesh.
    """

    return PartialGlobalMesh(global_mesh, lat_ind)


def extract_mesh_vector(partial_global_mesh: PartialGlobalMesh, lon_ind: int):
    """
    Extract one vector of values from the global mesh, i.e., nx x 1 x nz becomes 1 x 1 x nz.

    Parameters
    ----------
    partial_global_mesh : PartialGlobalMesh
        The partial global mesh containing the slice of the global grid.
    lon_ind : int
        The x index of the slice of the grid to be extracted.

    Returns
    -------
    MeshVector
        A struct containing one lat-lon point and the depths of all grid points at this location.
    """

    return MeshVector(partial_global_mesh, lon_ind)
