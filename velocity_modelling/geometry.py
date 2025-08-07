"""
Geometry Module for Velocity Model Construction

This module provides classes and functions for creating and manipulating 3D geometry
constructs needed for seismic velocity modelling. It handles coordinate transformations,
grid generation, mesh slicing, and boundary calculations for various model components.
"""

import logging
from logging import Logger
from typing import Any, Optional, Self

import numpy as np
from numba import njit

from velocity_modelling.constants import (
    DEP_GRID_DIM_MAX,
    EARTH_RADIUS_MEAN,
    ERAD,
    LAT_GRID_DIM_MAX,
    LON_GRID_DIM_MAX,
    MAX_LAT_SURFACE_EXTENSION,
    MAX_LON_SURFACE_EXTENSION,
    RPERD,
)


class GlobalMesh:
    """
    A global mesh of latitude, longitude, and depth points.

    Parameters
    ----------
    nx : int
        The number of points in the X direction.
    ny : int
        The number of points in the Y direction.
    nz : int
        The number of points in the Z direction.

    Attributes
    ----------
    lon : np.ndarray
        Array of longitudes.
    lat : np.ndarray
        Array of latitudes.
    max_lat : float
        Maximum latitude.
    min_lat : float
        Minimum latitude.
    max_lon : float
        Maximum longitude.
    min_lon : float
        Minimum longitude.
    x : np.ndarray (indexed by nx)
        Array of X coordinates.
    y : np.ndarray (indexed by ny)
        Array of Y coordinates.
    z : np.ndarray
        Array of Z coordinates. (in metres)

    """

    def __init__(self, nx: int, ny: int, nz: int):
        """
        Initialize a global mesh of latitude, longitude, and depth points.

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


class PartialGlobalMesh:
    """
    A partial mesh (nX x 1 x nZ) sliced from the full 3D global mesh (nX x nY x nZ)

    Parameters
    ----------
    global_mesh : GlobalMesh
        The model grid containing latitude, longitude, and depth points.
    lat_ind : int
        The latitude index of the slice to be extracted.

    Attributes
    ----------
    x : np.ndarray
        Array of X coordinates.
    y : np.ndarray
        Array of Y coordinates.
    z : np.ndarray
        Array of Z coordinates.
    lat : np.ndarray
        Array of latitudes.
    lon : np.ndarray
        Array of longitudes.

    """

    def __init__(self, global_mesh: GlobalMesh, lat_ind: int):
        """
        Create a partial mesh slice at a specific latitude index.

        """

        self.x = global_mesh.x.copy()
        self.y = global_mesh.y[lat_ind]
        self.z = global_mesh.z.copy()
        self.lat = global_mesh.lat[:, lat_ind].copy()
        self.lon = global_mesh.lon[:, lat_ind].copy()

    @property
    def nx(self) -> int:
        """
        Get the number of X points in the mesh.

        Returns
        -------
        int
            The number of X points.

        """
        return len(self.x)

    @property
    def nz(self) -> int:
        """
        Get the number of Z points in the mesh.

        Returns
        -------
        int
            The number of Z points.

        """
        return len(self.z)


class MeshVector:
    """
    A single mesh vector at a specific longitude index  from the partial global mesh.


    Parameters
    ----------
    partial_global_mesh : PartialGlobalMesh
        The partial mesh containing a slice in latitude.
    lon_ind : int
        Longitude index to extract from the partial global mesh.

    Attributes
    ----------
    lat : float
        Latitude of the vector.
    lon : float
        Longitude of the vector.
    z : np.ndarray
        Array of Z coordinates.
    vs30 : float
        Vs30 value.
    distance_from_shoreline : float
        Distance from the shoreline.

    """

    def __init__(self, partial_global_mesh: PartialGlobalMesh, lon_ind: int):
        """
        Create a single mesh vector at a specific longitude index.

        """

        self.lat = partial_global_mesh.lat[lon_ind]
        self.lon = partial_global_mesh.lon[lon_ind]
        self.z = partial_global_mesh.z.copy()
        self.vs30 = None
        self.distance_from_shoreline = None

    @property
    def nz(self):
        """
        Get the number of Z points in the mesh vector.

        Returns
        -------
        int
            The number of Z points.
        """
        return len(self.z)

    def copy(self) -> Self:
        """
        Create a deep copy of this MeshVector instance.

        Returns
        -------
        MeshVector
            A new MeshVector instance with the same attribute values.
        """
        # Create a new empty instance
        new_instance = object.__new__(MeshVector)

        # Copy scalar attributes
        new_instance.lat = self.lat
        new_instance.lon = self.lon

        # Deep copy array attributes
        if self.z is not None:
            new_instance.z = self.z.copy()
        else:
            new_instance.z = None

        # Copy optional attributes
        new_instance.vs30 = self.vs30
        new_instance.distance_from_shoreline = self.distance_from_shoreline

        return new_instance


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
        True if the point matches a boundary vertex within tolerance (1e-07), otherwise False.
    """
    # Vectorized comparison using NumPy with tolerance
    lat_matches = (
        np.abs(boundary_lats - lat) <= 1e-07
    )  # np.isclose is not supported by njit
    lon_matches = np.abs(boundary_lons - lon) <= 1e-07

    matches = np.logical_and(lat_matches, lon_matches)

    return np.any(matches)


@njit
def find_edge_inds(
    lats: np.ndarray,
    lons: np.ndarray,
    edge_type: int,
    max_lat: float,
    min_lat: float,
    max_lon: float,
    min_lon: float,
) -> tuple[int, int]:
    """
    Find the edge indices of the given latitude and longitude arrays.

    Parameters
    ----------
    lats : np.ndarray
        Latitude array.
    lons : np.ndarray
        Longitude array.
    edge_type : int
        Type of edge to find.(1: north, 2: east, 3: south, 4: west)
    max_lat : float
        Maximum latitude.
    min_lat : float
        Minimum latitude.
    max_lon : float
        Maximum longitude.
    min_lon : float
        Minimum longitude.

    Returns
    -------
    lat_edge_ind : int
        Index of the latitude edge.
    lon_edge_ind : int
        Index of the longitude edge

    """

    nlat = len(lats)
    nlon = len(lons)
    lat_edge_ind = 0
    lon_edge_ind = 0

    if edge_type == 1:  # north edge
        if max_lat == lats[0]:
            lat_edge_ind = 0
        elif max_lat == lats[-1]:
            lat_edge_ind = nlat - 1
        else:
            raise ValueError("Point lies outside of surface bounds.")
    elif edge_type == 3:  # south edge
        if min_lat == lats[0]:
            lat_edge_ind = 0
        elif min_lat == lats[-1]:
            lat_edge_ind = nlat - 1
        else:
            raise ValueError("Point lies outside of surface bounds.")
    elif edge_type == 2:  # east edge
        if max_lon == lons[0]:
            lon_edge_ind = 0
        elif max_lon == lons[-1]:
            lon_edge_ind = nlon - 1
        else:
            raise ValueError("Point lies outside of surface bounds.")
    elif edge_type == 4:  # west edge
        if min_lon == lons[0]:
            lon_edge_ind = 0
        elif min_lon == lons[-1]:
            lon_edge_ind = nlon - 1
        else:
            raise ValueError("Point lies outside of surface bounds.")
    else:
        raise ValueError("Point lies outside of surface bounds.")

    return lat_edge_ind, lon_edge_ind


@njit
def find_corner_inds(lats: np.ndarray, lons: np.ndarray, lat: float, lon: float):
    """
    Find the corner indices of the given latitude and longitude arrays.

    Parameters
    ----------
    lats : np.ndarray
        Latitude array.
    lons : np.ndarray
        Longitude array.
    lat : float
        Latitude of the given point.
    lon : float
        Longitude of the given point.

    Returns
    -------
    corner_lat_ind : int
        Index of the latitude corner.
    corner_lon_ind : int
        Index of the longitude corner

    """
    nlat = len(lats)
    nlon = len(lons)

    if np.abs(lat - lats[0]) <= 1e-07:
        corner_lat_ind = 0
    elif np.abs(lat - lats[-1]) <= 1e-07:
        corner_lat_ind = nlat - 1
    else:
        raise ValueError(
            f"Point lies outside of surface bounds. Lat {lat} outside [{lats[0]} {lats[-1]}]"
        )

    if np.abs(lon - lons[0]) <= 1e-07:
        corner_lon_ind = 0
    elif np.abs(lon - lons[-1]) <= 1e-07:
        corner_lon_ind = nlon - 1
    else:
        raise ValueError(
            f"Point lies outside of surface bounds. Lon {lon} outside [{lons[0]} {lons[-1]}]"
        )

    return corner_lat_ind, corner_lon_ind


@njit
def find_basin_adjacent_points_numba(
    lats: np.ndarray, lons: np.ndarray, lat: float, lon: float
) -> tuple[bool, np.ndarray, np.ndarray]:
    """
    Identify adjacent latitude/longitude indices for a given point in basin coordinates.

    Parameters
    ----------
    lats : np.ndarray
        Sorted array of latitudes (either ascending or descending).
    lons : np.ndarray
        Sorted array of longitudes (either ascending or descending).
    lat : float
        Target latitude for adjacency checks.
    lon : float
        Target longitude for adjacency checks.

    Returns
    -------
    bool
        True if the point is within surface bounds, otherwise False.
    np.ndarray of shape (2,)
        Latitude indices bounding the target point.
    np.ndarray of shape (2,)
        Longitude indices bounding the target point.
    """
    # Initialize default values
    in_surface_bounds = False
    lat_ind = np.array([0, 0], dtype=np.int64)
    lon_ind = np.array([0, 0], dtype=np.int64)

    # Handle latitude
    if lats[0] > lats[-1]:  # descending order
        lat_idx = np.searchsorted(lats[::-1], lat)
        lat_idx = len(lats) - lat_idx - 1
    else:  # ascending order
        lat_idx = np.searchsorted(lats, lat)

    if 0 < lat_idx < len(lats):
        lat_ind[0] = lat_idx - 1
        lat_ind[1] = lat_idx
    elif lat_idx == 0 and lats[0] == lat:
        lat_ind[0] = 0
        lat_ind[1] = 1
    elif lat_idx == len(lats) and lats[-1] == lat:
        lat_ind[0] = len(lats) - 1
        lat_ind[1] = len(lats)

    # Handle longitude
    if lons[0] > lons[-1]:  # descending order
        lon_idx = np.searchsorted(lons[::-1], lon)
        lon_idx = len(lons) - lon_idx - 1
    else:  # ascending order
        lon_idx = np.searchsorted(lons, lon)

    if 0 < lon_idx < len(lons):
        lon_ind[0] = lon_idx - 1
        lon_ind[1] = lon_idx
    elif lon_idx == 0 and lons[0] == lon:
        lon_ind[0] = 0
        lon_ind[1] = 1
    elif lon_idx == len(lons) and lons[-1] == lon:
        lon_ind[0] = len(lons) - 1
        lon_ind[1] = len(lons)

    if lat_ind[0] != 0 or lat_ind[1] != 0:
        if lon_ind[0] != 0 or lon_ind[1] != 0:
            in_surface_bounds = True

    return in_surface_bounds, lat_ind, lon_ind


@njit
def find_global_adjacent_points_numba(
    lats: np.ndarray, lons: np.ndarray, lat: float, lon: float
) -> tuple[
    bool, np.ndarray, np.ndarray, bool, int, int, bool, int, int, bool, int, int
]:
    """
    Identify adjacent indices and potential extension or corner zones for a global surface.

    Parameters
    ----------
    lats : np.ndarray
        Sorted array of latitudes (either ascending or descending).
    lons : np.ndarray
        Sorted array of longitudes (either ascending or descending).
    lat : float
        Target latitude for adjacency and zone checks.
    lon : float
        Target longitude for adjacency and zone checks.

    Returns
    -------
    bool
        True if the point is within normal surface bounds, otherwise False.
    np.ndarray of shape (2,)
        Latitude indices.
    np.ndarray of shape (2,)
        Longitude indices.
    bool
        True if in a latitude extension zone.
    int
        Latitude extension type code.
    int
        Latitude edge index in extension scenario.
    bool
        True if in a longitude extension zone.
    int
        Longitude extension type code.
    int
        Longitude edge index in extension scenario.
    bool
        True if in a corner extension scenario.
    int
        Corner latitude index.
    int
        Corner longitude index.
    """
    max_lat = np.max(lats)
    min_lat = np.min(lats)
    max_lon = np.max(lons)
    min_lon = np.min(lons)
    nlat = len(lats)
    nlon = len(lons)

    # Initialize default values
    in_surface_bounds = False
    lat_ind = np.array([0, 0], dtype=np.int64)
    lon_ind = np.array([0, 0], dtype=np.int64)
    in_lat_extension_zone = False
    lat_extension_type = 0
    lon_edge_ind = 0
    in_lon_extension_zone = False
    lon_extension_type = 0
    lat_edge_ind = 0
    in_corner_zone = False
    corner_lat_ind = 0
    corner_lon_ind = 0

    lat_assigned_flag = False
    lon_assigned_flag = False

    # Check if latitude is within the range
    if (lats[0] < lat <= lats[-1]) or (lats[0] > lat >= lats[-1]):
        is_ascending = lats[0] < lats[-1]
        # Create temp array for searching
        temp_lats = lats.copy()
        if not is_ascending:
            temp_lats = lats[::-1]

        index = np.searchsorted(temp_lats, lat)
        if not is_ascending:
            index = nlat - index

        if 0 < index < nlat:
            lat_ind[0] = index - 1
            lat_ind[1] = index
            lat_assigned_flag = True

    # Check if longitude is within the range
    if (lons[0] < lon <= lons[-1]) or (lons[0] > lon >= lons[-1]):
        is_ascending = lons[0] < lons[-1]
        # Create temp array for searching
        temp_lons = lons.copy()
        if not is_ascending:
            temp_lons = lons[::-1]

        index = np.searchsorted(temp_lons, lon)
        if not is_ascending:
            index = nlon - index

        if 0 < index < nlon:
            lon_ind[0] = index - 1
            lon_ind[1] = index
            lon_assigned_flag = True

    if lat_assigned_flag and lon_assigned_flag:
        in_surface_bounds = True
    else:
        # Handle extension zones
        if lon_assigned_flag and not lat_assigned_flag:
            if (lat - max_lat) <= MAX_LAT_SURFACE_EXTENSION and lat >= max_lat:
                in_lat_extension_zone = True
                lat_edge_ind, lon_edge_ind = find_edge_inds(
                    lats, lons, 1, max_lat, min_lat, max_lon, min_lon
                )
                lat_extension_type = 1
            elif (min_lat - lat) <= MAX_LAT_SURFACE_EXTENSION and lat <= min_lat:
                in_lat_extension_zone = True
                lat_edge_ind, lon_edge_ind = find_edge_inds(
                    lats, lons, 3, max_lat, min_lat, max_lon, min_lon
                )
                lat_extension_type = 3

        if lat_assigned_flag and not lon_assigned_flag:
            if (min_lon - lon) <= MAX_LON_SURFACE_EXTENSION and lon <= min_lon:
                in_lon_extension_zone = True
                lat_edge_ind, lon_edge_ind = find_edge_inds(
                    lats, lons, 4, max_lat, min_lat, max_lon, min_lon
                )
                lon_extension_type = 4
            elif (lon - max_lon) <= MAX_LON_SURFACE_EXTENSION and lon >= max_lon:
                in_lon_extension_zone = True
                lat_edge_ind, lon_edge_ind = find_edge_inds(
                    lats, lons, 2, max_lat, min_lat, max_lon, min_lon
                )
                lon_extension_type = 2

        # Handle corner zones
        corner_case = 0
        # Case 1: Top-left corner
        if (
            (lat - max_lat) <= MAX_LAT_SURFACE_EXTENSION
            and (min_lon - lon) <= MAX_LON_SURFACE_EXTENSION
            and lon <= min_lon
            and lat >= max_lat
        ):
            corner_lat_ind, corner_lon_ind = find_corner_inds(
                lats, lons, max_lat, min_lon
            )
            corner_case = 1
        # Case 2: Top-right corner
        elif (
            (lat - max_lat) <= MAX_LAT_SURFACE_EXTENSION
            and (lon - max_lon) <= MAX_LON_SURFACE_EXTENSION
            and lon >= max_lon
            and lat >= max_lat
        ):
            corner_lat_ind, corner_lon_ind = find_corner_inds(
                lats, lons, max_lat, max_lon
            )
            corner_case = 2
        # Case 3: Bottom-left corner
        elif (
            (min_lat - lat) <= MAX_LAT_SURFACE_EXTENSION
            and (min_lon - lon) <= MAX_LON_SURFACE_EXTENSION
            and lon <= min_lon
            and lat <= min_lat
        ):
            corner_lat_ind, corner_lon_ind = find_corner_inds(
                lats, lons, min_lat, min_lon
            )
            corner_case = 3
        # Case 4: Bottom-right corner
        elif (
            (min_lat - lat) <= MAX_LAT_SURFACE_EXTENSION
            and (lon - max_lon) <= MAX_LON_SURFACE_EXTENSION
            and lon >= max_lon
            and lat <= min_lat
        ):
            corner_lat_ind, corner_lon_ind = find_corner_inds(
                lats, lons, min_lat, max_lon
            )
            corner_case = 4

        in_corner_zone = corner_case > 0

    return (
        in_surface_bounds,
        lat_ind,
        lon_ind,
        in_lat_extension_zone,
        lat_extension_type,
        lon_edge_ind,
        in_lon_extension_zone,
        lon_extension_type,
        lat_edge_ind,
        in_corner_zone,
        corner_lat_ind,
        corner_lon_ind,
    )


class AdjacentPoints:
    """
    Store adjacent indices and flags for a given latitude/longitude point.

    Attributes
    ----------
    in_surface_bounds : bool
        True if the point is within the surface bounds.
    lat_ind : np.ndarray
        Latitude indices.
    lon_ind : np.ndarray
        Longitude indices.
    in_lat_extension_zone : bool
        True if the point is in a latitude extension zone.
    lat_extension_type : int
        Latitude extension type code.
    lon_edge_ind : int
        Longitude edge index in extension scenario.
    in_lon_extension_zone : bool
        True if the point is in a longitude extension zone.
    lon_extension_type : int
        Longitude extension type code.
    lat_edge_ind : int
        Latitude edge index in extension scenario.
    in_corner_zone : bool
        True if the point is in a corner extension scenario.
    corner_lat_ind : int
        Corner latitude index.
    corner_lon_ind : int
        Corner longitude index.

    Parameters
    ----------
    in_surface_bounds : bool
        True if the point is within the surface bounds.
    lat_ind : np.ndarray
        Latitude indices.
    lon_ind : np.ndarray
        Longitude indices.
    in_lat_extension_zone : bool
        True if the point is in a latitude extension zone.
    lat_extension_type : int
        Latitude extension type code.
    lon_edge_ind : int
        Longitude edge index in extension scenario.
    in_lon_extension_zone : bool
        True if the point is in a longitude extension zone.
    lon_extension_type : int
        Longitude extension type code.
    lat_edge_ind : int
        Latitude edge index in extension scenario.
    in_corner_zone : bool
        True if the point is in a corner extension scenario.
    corner_lat_ind : int
        Corner latitude index.
    corner_lon_ind : int
        Corner longitude index.


    """

    def __init__(
        self,
        in_surface_bounds: bool,
        lat_ind: np.ndarray,
        lon_ind: np.ndarray,
        in_lat_extension_zone: bool,
        lat_extension_type: int,
        lon_edge_ind: int,
        in_lon_extension_zone: bool,
        lon_extension_type: int,
        lat_edge_ind: int,
        in_corner_zone: bool,
        corner_lat_ind: int,
        corner_lon_ind: int,
    ):
        """
        Store bounding indices and flags to aid in interpolation checks.
        Use the find_basin_adjacent_points or find_global_adjacent_points class methods to populate.
        """
        self.in_surface_bounds = in_surface_bounds
        self.lat_ind = lat_ind
        self.lon_ind = lon_ind
        self.in_lat_extension_zone = in_lat_extension_zone
        self.lat_extension_type = lat_extension_type
        self.lon_edge_ind = lon_edge_ind
        self.in_lon_extension_zone = in_lon_extension_zone
        self.lon_extension_type = lon_extension_type
        self.lat_edge_ind = lat_edge_ind
        self.in_corner_zone = in_corner_zone
        self.corner_lat_ind = corner_lat_ind
        self.corner_lon_ind = corner_lon_ind

    @classmethod
    def find_basin_adjacent_points(
        cls, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float
    ) -> Self:
        """
        Find basin‐related adjacent indices and flags for a given latitude/longitude.

        Parameters
        ----------
        lats : np.ndarray
            Latitude array.
        lons : np.ndarray
            Longitude array.
        lat : float
            Target latitude.
        lon : float
            Target longitude.

        Returns
        -------
        AdjacentPoints
            Populated object with bounds data.
        """
        # Call the Numba function
        in_surface_bounds, lat_ind, lon_ind = find_basin_adjacent_points_numba(
            lats, lons, lat, lon
        )

        if not in_surface_bounds:
            raise ValueError(
                f"Basin point lies outside of the extent of the basin surface . {lon} {lat}"
            )

        # Create the AdjacentPoints object with defaults for unused fields
        return cls(
            in_surface_bounds=in_surface_bounds,
            lat_ind=lat_ind,
            lon_ind=lon_ind,
            in_lat_extension_zone=False,
            lat_extension_type=0,
            lon_edge_ind=0,
            in_lon_extension_zone=False,
            lon_extension_type=0,
            lat_edge_ind=0,
            in_corner_zone=False,
            corner_lat_ind=0,
            corner_lon_ind=0,
        )

    @classmethod
    def find_global_adjacent_points(
        cls, lats: np.ndarray, lons: np.ndarray, lat: float, lon: float
    ) -> Self:
        """
        Find the adjacent points to the given latitude and longitude in the global surface.
        This is a wrapper around the Numba-optimized function.

        Parameters
        ----------
        lats : np.ndarray
            Array of latitudes of the global surface.
        lons : np.ndarray
            Array of longitudes of the global surface.
        lat : float
            Latitude of the point to check.
        lon : float
            Longitude of the point to check.

        Returns
        -------
        AdjacentPoints
            Object containing the adjacent points.
        """
        # Call the Numba function
        result = find_global_adjacent_points_numba(lats, lons, lat, lon)
        # Unpack the result and create an AdjacentPoints object
        adjacent_points = AdjacentPoints(*result)

        # Raise error if needed
        if not (
            adjacent_points.in_surface_bounds
            or adjacent_points.in_lat_extension_zone
            or adjacent_points.in_lon_extension_zone
            or adjacent_points.in_corner_zone
        ):
            raise ValueError(
                f"Point does not lie in any global surface extension. {lon} {lat}"
            )

        return adjacent_points


class SmoothingBoundary:
    """
    Handle boundary smoothing between points.

    Attributes
    ----------
    lons : list[float]
        Longitude coordinates.
    lats : list[float]
        Latitude coordinates.
    n_points : int
        Number of points in the boundary

    Parameters
    ----------
    lons : list[float]
        longitude coordinates.
    lats : list[float]
        latitude coordinates.
    """

    def __init__(self, lons: list[float], lats: list[float]):
        """
        Handle boundary smoothing between points.

        """
        self.lons = lons
        self.lats = lats
        self.n_points = len(lons)
        if len(self.lons) != len(self.lats):
            raise ValueError(
                f"Longitude length does not match latitude length ({len(self.lons)} != {len(self.lats)}"
            )

    def determine_if_lat_lon_within_smoothing_region(
        self, mesh_vector: MeshVector
    ) -> tuple[int, float]:
        """
        Find the closest boundary index and distance to the given mesh vector coordinates.

        Parameters
        ----------
        mesh_vector : MeshVector
            The mesh vector with lat/lon.

        Returns
        -------
        int
            Index of the closest boundary segment.
        float
            Distance to that boundary in kilometers.
        """
        closest_ind, distance = self.brute_force(mesh_vector)

        return closest_ind, distance / 1000  # return distance in km

    def brute_force(self, mesh_vector: MeshVector) -> tuple[int, float]:
        """
        Determine the closest index within the smoothing boundary to the given mesh vector coordinates.

        Parameters
        ----------
        mesh_vector : MeshVector
            Object containing mesh vector data.

        Returns
        -------
        int
            Index of the closest boundary segment.
        float
            Distance to that boundary

        """
        boundary_points = np.column_stack((self.lats, self.lons))
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

    d_lon = lon_rad - ref_lon

    dz = np.sin(lat_rad) - np.sin(ref_lat)
    dx = np.cos(d_lon) * np.cos(lat_rad) - np.cos(ref_lat)
    dy = np.sin(d_lon) * np.cos(lat_rad)

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
    erad : float
        Earth's radius (default is ERAD).
    g0 : float
        Initial longitude (default is 0).
    b0 : float
        Initial latitude (default is 0).

    Returns
    -------
    tuple[np.ndarray, Any]
        Computed latitude and longitude arrays.
    """
    cos_b = np.cos(x / erad - b0)
    sin_b = np.sin(x / erad - b0)

    cos_g = np.cos(y / erad - g0)
    sin_g = np.sin(y / erad - g0)

    xp = sin_g * cos_b * np.sqrt(1 + sin_b * sin_b * sin_g * sin_g)
    yp = sin_b * cos_g * np.sqrt(1 + sin_b * sin_b * sin_g * sin_g)
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
    vm_params: dict, logger: Optional[Logger] = None
) -> GlobalMesh:
    """
    Generate a global mesh grid using great-circle projection.

    Parameters
    ----------
    vm_params : dict
        Defines the model's dimensions and origin.
    logger : Logger, optional
        Logger instance for reporting progress.

    Returns
    -------
    GlobalMesh
        The generated global mesh object.
    """
    if logger is None:
        logger = Logger(name="velocity_model.geometry")

    logger.log(logging.INFO, "Starting generation of model grid.")

    xmax = vm_params["extent_x"]
    ymax = vm_params["extent_y"]
    zmax = vm_params["extent_zmax"]
    zmin = vm_params["extent_zmin"]
    h_depth = vm_params["h_depth"]
    h_lat_lon = vm_params["h_lat_lon"]
    nx = vm_params["nx"]
    ny = vm_params["ny"]
    nz = vm_params["nz"]
    origin_lat = vm_params["origin_lat"]
    origin_lon = vm_params["origin_lon"]
    origin_rot = vm_params["origin_rot"]

    global_mesh = GlobalMesh(nx, ny, nz)

    global_mesh.max_lat = -180
    global_mesh.min_lat = 0
    global_mesh.max_lon = 0
    global_mesh.min_lon = 180

    # Check if the grid dimensions are consistent with the provided parameters
    # Python round() has different behavior than C round() and rounds to the closest even number round(2.5)->2, round(3.5)->4
    # Use adding 0.5 then casting with int() to achieve round-half-up behavior, which matches the intended calculation method.
    nx_expected = int(xmax / h_lat_lon + 0.5)
    ny_expected = int(ymax / h_lat_lon + 0.5)
    nz_expected = int((zmax - zmin) / h_depth + 0.5)

    if nx != nx_expected:
        raise ValueError(
            f" nx: {nx} !=  xmax: {xmax} / h_lat_lon: {h_lat_lon}(={nx_expected})"
        )
    if ny != ny_expected:
        raise ValueError(
            f" ny: {ny} !=  ymax: {ymax} / h_lat_lon: {h_lat_lon}(={ny_expected})"
        )
    if nz != nz_expected:
        raise ValueError(
            f" nz: {nz} !=  (zmax-zmin: {zmax}-{zmin}) / h_depth: {h_depth})(={nz_expected})"
        )

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
        logger.log(
            logging.INFO, f"Number of model points. nx: {nx}, ny: {ny}, nz: {nz}."
        )

    global_mesh.x = 0.5 * h_lat_lon + h_lat_lon * np.arange(nx) - 0.5 * xmax

    global_mesh.y = 0.5 * h_lat_lon + h_lat_lon * np.arange(ny) - 0.5 * ymax

    global_mesh.z = -1000 * (0.5 * h_depth + h_depth * np.arange(nz) + zmin)

    arg = origin_rot * RPERD
    cos_a = np.cos(arg)
    sin_a = np.sin(arg)

    arg = (90.0 - origin_lat) * RPERD
    cos_t = np.cos(arg)
    sin_t = np.sin(arg)

    arg = origin_lon * RPERD
    cos_p = np.cos(arg)
    sin_p = np.sin(arg)

    amat = np.array(
        [
            cos_a * cos_t * cos_p + sin_a * sin_p,
            sin_a * cos_t * cos_p - cos_a * sin_p,
            sin_t * cos_p,
            cos_a * cos_t * sin_p - sin_a * cos_p,
            sin_a * cos_t * sin_p + cos_a * cos_p,
            sin_t * sin_p,
            -cos_a * sin_t,
            -sin_a * sin_t,
            cos_t,
        ]
    ).reshape((3, 3))

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

    logger.log(logging.INFO, "Completed Generation of Model Grid.")
    return global_mesh
