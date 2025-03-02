import numpy as np
from logging import Logger
import logging

import sys

from velocity_modelling.cvm.registry import CVMRegistry
from velocity_modelling.cvm.interpolate import bi_linear_interpolation
from velocity_modelling.cvm.geometry import (
    point_on_vertex,
    AdjacentPoints,
    MeshVector,
    GlobalMesh,
    PartialGlobalMesh,
    extract_partial_mesh,
    SmoothingBoundary,
)

from qcore import point_in_polygon
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains
from shapely.strtree import STRtree
from typing import List, Tuple


def check_boundary_index(func):
    def wrapper(self, i, *args, **kwargs):
        if i < 0 or i >= len(self.boundaries):
            self.log(
                f"Error: basin boundary {i} not found. Max index is {len(self.boundaries) - 1}"
            )
            return None
        return func(self, i, *args, **kwargs)

    return wrapper


class BasinData:
    def __init__(
        self, cvm_registry: CVMRegistry, basin_name: str, logger: Logger = None
    ):
        """
        Initialize the BasinData.

        Parameters
        ----------
        cvm_registry : CVMRegistry
            The CVMRegistry instance.
        basin_name : str
            The name of the basin.
        logger : Logger, optional
            The logger instance.
        """
        self.name = basin_name

        basin_info = cvm_registry.get_info("basin", basin_name)

        self.boundaries = [
            cvm_registry.load_basin_boundary(boundary_path)
            for boundary_path in basin_info["boundaries"]
        ]
        self.surfaces = [
            cvm_registry.load_basin_surface(surface)
            for surface in basin_info["surfaces"]
        ]
        self.submodels = [
            cvm_registry.load_basin_submodel(surface)
            for surface in basin_info["surfaces"]
        ]

        self.perturbation_data = None
        self.logger = logger

        self.log(f"Basin {basin_name} fully loaded.")

    def log(self, message, level=logging.INFO):
        """
        Log a message.

        Parameters
        ----------
        message : str
            The message to log.
        level : int, optional
            The logging level.
        """
        if self.logger is not None:
            self.logger.log(level, message)
        else:
            print(message, file=sys.stderr)

    @check_boundary_index
    def boundary_lat(self, i: int) -> np.ndarray:
        """
        Get the latitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        np.ndarray
            The latitude of the boundary.
        """
        return self.boundaries[i][:, 1]

    @check_boundary_index
    def boundary_lon(self, i: int) -> np.ndarray:
        """
        Get the longitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        np.ndarray
            The longitude of the boundary.
        """
        return self.boundaries[i][:, 0]


class InBasinGlobalMesh:
    def __init__(
        self,
        global_mesh: GlobalMesh,
        basin_data_list: List[BasinData],
        smooth_bound: SmoothingBoundary = None,
    ):
        """
        Initialize with basin membership for global_mesh and optionally smooth_bound.

        Parameters
        ----------
        global_mesh : GlobalMesh
            The global mesh containing lat/lon values.
        basin_data_list : List[BasinData]
            List of BasinData objects for basin membership.
        smooth_bound : SmoothingBoundary, optional
            Smoothing boundary with x (lon) and y (lat) arrays.
        """
        self.nx, self.ny = global_mesh.lat.shape
        self.nz = len(global_mesh.z)
        # Use object dtype to store lists of basin indices
        # Initialize basin_membership as an (ny, nx) array of empty lists
        self.basin_membership = [[[] for _ in range(self.nx)] for _ in range(self.ny)]
        self.basin_data_list = basin_data_list  # Store for preprocess_smooth_bound
        self.smooth_basin_membership = None  # For smooth_bound points
        if smooth_bound is not None:
            print(f"DEBUG: smooth_bound provided, n={smooth_bound.n}")  # Debug
            self.preprocess_smooth_bound(smooth_bound)
        else:
            print("DEBUG: smooth_bound is None")

    def preprocess_smooth_bound(self, smooth_bound: SmoothingBoundary):
        print(
            f"DEBUG: Preprocessing smooth_bound with {smooth_bound.n} points"
        )  # Temporary debug
        n_points = smooth_bound.n
        self.smooth_basin_membership = [[] for _ in range(n_points)]
        for i in range(n_points):
            lat = smooth_bound.y[i]
            lon = smooth_bound.x[i]
            for basin_idx, basin_data in enumerate(self.basin_data_list):
                if determine_if_within_basin_lat_lon(basin_data, lat, lon):
                    self.smooth_basin_membership[i].append(basin_idx)
        print(
            f"DEBUG: smooth_basin_membership initialized with length {len(self.smooth_basin_membership)}"
        )  # Temporary debug


def determine_if_within_basin_lat_lon(basin_data: BasinData, lat: float, lon: float):
    """
    Determine if a point lies within the different basin boundaries.

    Parameters
    ----------
    mesh_vector : MeshVector
        Struct containing a single lat-lon point.

    Returns
    -------
    bool
        True if inside a basin, False otherwise.
    """

    # TODO: Only Perturbation basins are ignored for smoothing, which will be handled in the perturbation code.
    #  We dropped ignoreBasinForSmoothing from Basin definition. By default we don't ignore any basins for smoothing.

    # See https://github.com/ucgmsim/mapping/blob/80b8e66222803d69e2f8f2182ccc1adc467b7cb1/mapbox/vs30/scripts/basin_z_values/gen_sites_in_basin.py#L119C2-L123C55
    # and https://github.com/ucgmsim/qcore/blob/master/qcore/point_in_polygon.py

    for ind, boundary in enumerate(basin_data.boundaries):

        if not (
            np.min(boundary[:, 0]) <= lon <= np.max(boundary[:, 0])
            and np.min(boundary[:, 1]) <= lat <= np.max(boundary[:, 1])
        ):
            continue  # outside of basin

        else:
            # possibly in basin
            in_poly = point_in_polygon.is_inside_postgis(
                boundary, np.array([lon, lat])
            )  # check if in poly
            if in_poly:  # in_poly == 1 (inside) ==2 (on edge)
                return True  # inside a basin (any)
            else:  # outside poly
                # check if it is on vertex. if in_poly
                on_vertex = point_on_vertex(
                    basin_data.boundary_lat(ind),
                    basin_data.boundary_lon(ind),
                    lat,
                    lon,
                )
                if on_vertex:
                    return True
                continue  # outside of basin

    return False  # not inside basin


class InBasin:
    def __init__(self, basin_data: BasinData, n_depths: int):
        """
        Initialize the InBasin. Used to determine if a given point is within the basin.

        Parameters
        ----------
        basin_data : BasinData
            The BasinData instance.
        n_depths : int
            The number of depth points.
        """
        self.basin_data = basin_data
        # the given lat lon is within a boundary of this basin
        self.in_basin_lat_lon = False
        # checks the basin's surface depth and if the point in the range of n_depths is within the basin
        self.in_basin_depth = np.full((n_depths), False, dtype=bool)


def preprocess_basin_membership(
    global_mesh: GlobalMesh,
    basin_data_list: List[BasinData],
    logger: Logger,
    smooth_bound: SmoothingBoundary = None,
) -> Tuple[InBasinGlobalMesh, List[PartialGlobalMesh]]:
    logger.info(f"DEBUG: smooth_bound in preprocess: {smooth_bound}")
    in_basin_mesh = InBasinGlobalMesh(global_mesh, basin_data_list, smooth_bound)
    nx, ny = in_basin_mesh.nx, in_basin_mesh.ny
    partial_global_mesh_list = [extract_partial_mesh(global_mesh, j) for j in range(ny)]
    logger.info(f"Pre-processing basin membership for {len(basin_data_list)} basins.")

    for j in range(ny):
        partial_global_mesh = partial_global_mesh_list[j]
        for k in range(nx):
            lat = partial_global_mesh.lat[k]
            lon = partial_global_mesh.lon[k]
            for basin_idx, basin_data in enumerate(basin_data_list):
                if determine_if_within_basin_lat_lon(basin_data, lat, lon):
                    in_basin_mesh.basin_membership[j][k].append(basin_idx)

    if smooth_bound is not None:
        logger.info(
            f"Pre-processed smooth boundary membership for {smooth_bound.n} points."
        )
    logger.info(
        f"DEBUG: in_basin_mesh.smooth_basin_membership after preprocess: {in_basin_mesh.smooth_basin_membership}"
    )
    return (in_basin_mesh, partial_global_mesh_list)


class PartialBasinSurfaceDepths:
    def __init__(self, basin_data: BasinData):
        """
        Initialize the PartialBasinSurfaceDepths.

        Parameters
        ----------
        basin_data : BasinData
            The BasinData instance.
        """
        # List of arrays of depths for each surface of the basin
        # self.depths[i] is the depth of the i-th surfaces
        self.depths = np.full(len(basin_data.surfaces), np.nan, dtype=np.float64)
        self.basin = basin_data

    def determine_basin_surface_depths(
        self,
        inbasin: InBasin,
        mesh_vector: MeshVector,
    ):
        """
        Determine the basin surface depths for a given latitude and longitude.

        Parameters
        ----------
        inbasin : InBasin
            Struct containing flags to indicate if lat-lon point - depths lie within the basin.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        """
        assert inbasin.in_basin_lat_lon
        # this is only executed if inbasin.in_basin_lat_lon is True.

        for surface_ind, surface in enumerate(inbasin.basin_data.surfaces):
            adjacent_points = AdjacentPoints.find_basin_adjacent_points(
                surface.lati, surface.loni, mesh_vector.lat, mesh_vector.lon
            )
            # TODO: check if in_surface_bounds is True
            x1 = surface.loni[adjacent_points.lon_ind[0]]
            x2 = surface.loni[adjacent_points.lon_ind[1]]
            y1 = surface.lati[adjacent_points.lat_ind[0]]
            y2 = surface.lati[adjacent_points.lat_ind[1]]
            q11 = surface.raster[adjacent_points.lon_ind[0]][adjacent_points.lat_ind[0]]
            q12 = surface.raster[adjacent_points.lon_ind[0]][adjacent_points.lat_ind[1]]

            q21 = surface.raster[adjacent_points.lon_ind[1]][adjacent_points.lat_ind[0]]

            q22 = surface.raster[adjacent_points.lon_ind[1]][adjacent_points.lat_ind[1]]
            self.depths[surface_ind] = bi_linear_interpolation(
                x1,
                x2,
                y1,
                y2,
                q11,
                q12,
                q21,
                q22,
                mesh_vector.lon,
                mesh_vector.lat,
            )

    # TODO: can be inserted into enforce_basin_surface_depths()
    def enforce_surface_depths(self):
        """
        Enforce the depths of the surface are consistent with stratigraphy.

        Returns
        -------
        None
            Updates the depths in place.
        """

        # Find the first NaN value
        nan_indices = np.where(np.isnan(self.depths))[0]
        if nan_indices.size > 0:
            nan_ind = nan_indices[0]
        else:
            nan_ind = len(self.depths)
        # Enforce stratigraphy for depths before the first NaN
        for i in range(nan_ind - 1, 0, -1):
            if self.depths[i - 1] < self.depths[i]:
                self.depths[i - 1] = self.depths[i]
        # Enforce stratigraphy for depths after the first NaN
        for i in range(nan_ind - 1):
            if self.depths[i] < self.depths[i + 1]:
                self.depths[i] = self.depths[i + 1]

    def determine_basin_surface_above(self, depth: float):
        """
        Determine the index of the basin surface directly above the given depth.

        Parameters
        ----------
        depth : float
            The depth of the grid point to determine the properties at.

        Returns
        -------
        int
            Index of the surface directly above the grid point.
        """
        # self.depths is in decreasing order

        valid_indices = np.where((~np.isnan(self.depths)) & (self.depths >= depth))[0]
        return valid_indices[-1] if valid_indices.size > 0 else 0  # the last one

    def determine_basin_surface_above_vectorized(self, depths: np.ndarray):
        """
        Determine the indices of the basin surfaces directly above the given depths.

        Parameters
        ----------
        depths : np.ndarray
            Array of depths for multiple grid points.

        Returns
        -------
        np.ndarray
            Array of indices, each corresponding to the surface directly above the respective depth.
        """
        # Ensure depths is a NumPy array
        depths = np.asarray(depths)

        # Initialize output array with default value 0
        indices = np.zeros_like(depths, dtype=int)

        # Mask for valid depths in self.depths (not NaN)
        valid_mask = ~np.isnan(self.depths)
        valid_depths = self.depths[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        if valid_depths.size == 0:
            return indices  # All depths are invalid, return zeros

        # For each depth, find the last valid index where self.depths >= depth
        # Since self.depths is in decreasing order, we can use searchsorted
        # searchsorted finds the insertion point where depth would be inserted to maintain order
        # Since depths are decreasing, we want the rightmost index where valid_depths >= depth
        search_indices = np.searchsorted(-valid_depths, -depths, side="right")

        # Adjust indices to account for valid_mask
        valid_search_indices = np.clip(search_indices - 1, 0, valid_depths.size - 1)
        indices = valid_indices[valid_search_indices]

        # If no valid depths are >= depth, return 0 (as in original logic)
        indices[search_indices == 0] = 0

        return indices

    def determine_basin_surface_below(self, depth: float):
        """
        Determine the index of the basin surface directly below the given depth.

        Parameters
        ----------
        depth : float
            The depth of the grid point to determine the properties at.

        Returns
        -------
        int
            Index of the surface directly below the grid point.
        """
        valid_indices = np.where((~np.isnan(self.depths)) & (self.depths <= depth))[0]
        return valid_indices[-1] if valid_indices.size > 0 else 0  # the last index

    def enforce_basin_surface_depths(
        self,
        in_basin: InBasin,
        mesh_vector: MeshVector,
    ):
        """
        Enforce the depths of the surfaces are consistent with stratigraphy.

        Parameters
        ----------
        in_basin : InBasin
            Struct containing flags to indicate if lat-lon point - depths lie within the basin.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        """
        assert in_basin.in_basin_lat_lon

        self.enforce_surface_depths()
        # TODO: check if this is correct
        top_lim = self.depths[0]  # the depth of the first surface
        bot_lim = self.depths[-1]  # the depth of the last surface

        in_basin.in_basin_depth = (bot_lim <= mesh_vector.z) & (
            mesh_vector.z <= top_lim
        )  # check if the point is within the basin

    def interpolate_basin_surface_depths(
        self,
        in_basin: InBasin,
        mesh_vector: MeshVector,
    ):
        """
        Determine if a lat-lon point is in a basin, if so interpolate the basin surface depths, enforce their hierarchy,
         then determine which depth points lie within the basin limits.

        Parameters
        ----------
        in_basin : InBasin
            Struct containing flags to indicate if lat-lon point - depths lie within the basin.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        """
        self.determine_basin_surface_depths(in_basin, mesh_vector)
        self.enforce_basin_surface_depths(in_basin, mesh_vector)


class BasinSurfaceRead:
    def __init__(self, nlat: int, nlon: int):
        """
        Initialize the BasinSurfaceRead.

        Parameters
        ----------
        nlat : int
            The number of latitude points.
        nlon : int
            The number of longitude points.
        """
        self.lati = np.zeros(nlat)
        self.loni = np.zeros(nlon)
        self.raster = np.zeros((nlon, nlat))
        self.max_lat = None
        self.min_lat = None
        self.max_lon = None
        self.min_lon = None
