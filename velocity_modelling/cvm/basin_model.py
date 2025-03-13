"""
Basin Model Module

This module manages basin data representation, surface interpolation, and basin membership
determination in the velocity model. It handles basin boundaries, surfaces, and submodels
with proper logging throughout the processing workflow.
"""

import logging
from logging import Logger
from typing import Self

import numpy as np
from numba import njit

from qcore import point_in_polygon
from velocity_modelling.cvm.geometry import (
    AdjacentPoints,
    GlobalMesh,
    MeshVector,
    PartialGlobalMesh,
    SmoothingBoundary,
    extract_partial_mesh,
    point_on_vertex,
)
from velocity_modelling.cvm.interpolate import (
    bi_linear_interpolation,
)
from velocity_modelling.cvm.registry import CVMRegistry


class BasinData:
    """
    Container for basin data, including boundaries, surfaces, and submodels.

    Parameters
    ----------
    cvm_registry : CVMRegistry
        The CVMRegistry instance.
    basin_name : str
        The name of the basin.
    logger : Logger, optional
        The logger instance.

    Attributes
    ----------
    name : str
        Name of the basin.
    boundaries : list of np.ndarray
        List of basin boundaries.
    surfaces : list of np.ndarray
        List of basin surfaces.
    submodels : list of np.ndarray
        List of basin submodels.
    perturbation_data : None
        Placeholder for perturbation data.

    """

    def __init__(
        self, cvm_registry: CVMRegistry, basin_name: str, logger: Logger = None
    ):
        """
        Initialize the BasinData object.

        """

        if logger is None:
            self.logger = Logger(name="velocity_model.basin_data")
        else:
            self.logger = logger

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

        self.logger.log(logging.INFO, f"Basin {basin_name} fully loaded.")


@njit
def determine_basin_contains_lat_lon(
    boundaries: list[np.ndarray], lat: float, lon: float
):
    """
    Check if a given (lat, lon) lies within or on a vertex of any basin boundary.

    Parameters
    ----------
    boundaries : list of np.ndarray
        list of arrays representing basin boundaries.
    lat : float
        Latitude to check.
    lon : float
        Longitude to check.

    Returns
    -------
    bool
        True if the point is inside or on a vertex of a boundary, otherwise False.
    """
    # TODO: Only Perturbation basins are ignored for smoothing, which will be handled in the perturbation code.
    #  We dropped ignoreBasinForSmoothing from Basin definition. By default we don't ignore any basins for smoothing.

    # Filtering by bounding box has been already done
    for ind in range(len(boundaries)):
        boundary = boundaries[ind]

        lats = boundary[:, 1]
        lons = boundary[:, 0]

        if point_in_polygon.is_inside_postgis(boundary, (lon, lat)) or point_on_vertex(
            lats, lons, lat, lon
        ):
            return True

    return False


class InBasinGlobalMesh:
    """
    Tracks basin membership for each (lat,lon) point of the global mesh, optionally
    including a smoothing boundary.
    Instances should be created using the preprocess_basin_membership() class method.

    Parameters
    ----------
    global_mesh : GlobalMesh
        The global mesh containing lat/lon values.
    basin_data_list : list[BasinData]
        List of BasinData objects for basin membership.
    logger : Logger, Optional
        Logger instance.

    Attributes
    ----------
    nx : int
        Number of x-coordinates in the global mesh.
    ny : int
        Number of y-coordinates in the global mesh.
    nz : int
        Number of z-coordinates in the global mesh.
    logger : Logger
        Logger instance.
    basin_data_list : list[BasinData]
        List of BasinData objects for basin membership.
    smooth_basin_membership : list[list[int]]
        List of basin indices for each point in the smoothing boundary.
    basin_membership : list[list[list[int]]]
        List of basin indices for each point in the global mesh.
    min_basin_boundary_lons : np.ndarray
        Minimum longitude for each basin.
    max_basin_boundary_lons : np.ndarray
        Maximum longitude for each basin.
    min_basin_boundary_lats : np.ndarray
        Minimum latitude for each basin.
    max_basin_boundary_lats : np.ndarray
        Maximum latitude for each basin

    """

    def __init__(
        self,
        global_mesh: GlobalMesh,
        basin_data_list: list[BasinData],
        logger: Logger = None,
    ):
        """
        Private constructor. Use preprocess_basin_membership() instead to create instances.

        """
        if logger is None:
            self.logger = Logger(name="velocity_model.in_basin_global_mesh")
        else:
            self.logger = logger

        self.nx, self.ny = global_mesh.lat.shape
        self.nz = len(global_mesh.z)

        self.basin_data_list = basin_data_list
        self.smooth_basin_membership = None
        self.basin_membership = None  # Will be set by preprocess_basin_membership

        # Convert all basin boundaries into NumPy arrays
        self.min_basin_boundary_lons = self.max_basin_boundary_lons = (
            self.min_basin_boundary_lats
        ) = self.max_basin_boundary_lats = None

    @classmethod
    def preprocess_basin_membership(
        cls,
        global_mesh: GlobalMesh,
        basin_data_list: list[BasinData],
        logger: Logger = None,
        smooth_bound: SmoothingBoundary = None,
    ) -> tuple[Self, list[PartialGlobalMesh]]:
        """
        Preprocess basin membership for a given global mesh to speed up the velocity model generation
        This method is the recommended way to create an InBasinGlobalMesh object.

        Parameters
        ----------
        global_mesh : GlobalMesh
            Global mesh where each (x, y) is a lat-lon point.
        basin_data_list : list of BasinData
            Collection of BasinData objects.
        logger : Logger, optional
            Logger for status reporting.
        smooth_bound : SmoothingBoundary, optional
            Optional boundary for smoothing.

        Returns
        -------
        tuple of (InBasinGlobalMesh, list of PartialGlobalMesh)
            Mesh membership object and list of partial slices.
        """
        if logger is None:
            logger = Logger(name="velocity_model.in_basin_global_mesh")

        in_basin_mesh = cls(global_mesh, basin_data_list, logger)

        # Use object dtype to store lists of basin indices
        # Initialize basin_membership as an (ny, nx) array of empty lists
        in_basin_mesh.basin_membership = [
            [[] for _ in range(in_basin_mesh.nx)] for _ in range(in_basin_mesh.ny)
        ]

        boundary_arrays = [
            np.vstack(basin.boundaries)  # Merge all boundary arrays for each basin
            for basin in basin_data_list
        ]

        # Compute min/max lat/lon per basin
        in_basin_mesh.min_basin_boundary_lons = np.array(
            [np.min(boundary[:, 0]) for boundary in boundary_arrays]
        )
        in_basin_mesh.max_basin_boundary_lons = np.array(
            [np.max(boundary[:, 0]) for boundary in boundary_arrays]
        )
        in_basin_mesh.min_basin_boundary_lats = np.array(
            [np.min(boundary[:, 1]) for boundary in boundary_arrays]
        )
        in_basin_mesh.max_basin_boundary_lats = np.array(
            [np.max(boundary[:, 1]) for boundary in boundary_arrays]
        )

        if smooth_bound is not None:
            logger.log(
                logging.DEBUG,
                f"Initializing smooth boundary with {smooth_bound.n_points} points",
            )
            in_basin_mesh.preprocess_smooth_bound(smooth_bound)
            logger.log(
                logging.DEBUG,
                f"Pre-processed smooth boundary membership for {smooth_bound.n_points} points.",
            )
            logger.log(
                logging.DEBUG,
                f"in_basin_mesh.smooth_basin_membership after preprocess: {in_basin_mesh.smooth_basin_membership}",
            )
        else:
            logger.log(logging.DEBUG, "smooth_bound is None")

        nx, ny = in_basin_mesh.nx, in_basin_mesh.ny
        partial_global_mesh_list = [
            extract_partial_mesh(global_mesh, j) for j in range(ny)
        ]

        for j in range(ny):
            partial_global_mesh = partial_global_mesh_list[j]
            for k in range(nx):
                lat = partial_global_mesh.lat[k]
                lon = partial_global_mesh.lon[k]
                in_basin_mesh.basin_membership[j][k] = (
                    in_basin_mesh.find_all_containing_basins(lat, lon)
                )

        logger.log(
            logging.INFO,
            f"Pre-processed basin membership for {len(basin_data_list)} basins.",
        )
        return (in_basin_mesh, partial_global_mesh_list)

    def get_basin_membership(self, x: int, y: int) -> list[int]:
        """
        Get the basin membership for a given (x, y) point.

        Parameters
        ----------
        x : int
            The index in x-direction.
        y : int
            The index in y-direction.

        Returns
        -------
        list[int]
            Indices of basins containing the point.
        """
        if self.smooth_basin_membership is None:
            raise ValueError("Smooth basin membership not pre-processed.")
        return self.basin_membership[y][x]

    def find_all_containing_basins(self, lat: float, lon: float) -> list[int]:
        """
        Determine all basins that contain a given (lat, lon).
        Use this if basin_membership is not available.

        Parameters
        ----------
        lat : float
            Latitude for the point.
        lon : float
            Longitude for the point.

        Returns
        -------
        list[int]
            Indices of basins containing the point.
        """
        # Step 1: Vectorized Bounding Box Check
        inside_bbox = (
            (self.min_basin_boundary_lons <= lon)
            & (lon <= self.max_basin_boundary_lons)
            & (self.min_basin_boundary_lats <= lat)
            & (lat <= self.max_basin_boundary_lats)
        )

        # Get candidate basin indices
        candidate_indices = np.where(inside_bbox)[0]

        # Step 2: Polygon Check (Only for Candidates)
        # TODO: could be further optimized boundary_arrays (in preprocess_basin_membership()) is used
        return [
            idx
            for idx in candidate_indices
            if determine_basin_contains_lat_lon(
                self.basin_data_list[idx].boundaries, lat, lon
            )
        ]  # Returns all matching basin indices

    def preprocess_smooth_bound(self, smooth_bound: SmoothingBoundary):
        """
        Precompute basin membership for smoothing boundary points.

        Parameters
        ----------
        smooth_bound : SmoothingBoundary
            Boundary with 'lons' and 'lats' coordinate arrays and an integer 'n_points'.
        """
        self.logger.log(
            logging.DEBUG,
            f"Preprocessing smooth_bound with {smooth_bound.n_points} points",
        )

        n_points = smooth_bound.n_points
        self.smooth_basin_membership = [[] for _ in range(n_points)]

        for i in range(n_points):
            lat = smooth_bound.lats[i]
            lon = smooth_bound.lons[i]
            self.smooth_basin_membership[i] = self.find_all_containing_basins(lat, lon)

        self.logger.log(
            logging.DEBUG,
            f"smooth_basin_membership initialized with length {len(self.smooth_basin_membership)}",
        )


class InBasin:
    """
    Tracks if a point is within a basin and which depths apply.

    Parameters
    ----------
    basin_data : BasinData
        The BasinData instance.
    n_depths : int
        Number of depth points in the mesh.

    Attributes
    ----------
    basin_data : BasinData
        The BasinData instance.
    in_basin_lat_lon : bool
        True if the lat-lon point lies within the basin's boundaries.
    in_basin_depth : np.ndarray
        True if the depth point lies within the basin's surfaces.
    """

    def __init__(self, basin_data: BasinData, n_depths: int):
        """
        Initialize the InBasin object.

        """

        self.basin_data = basin_data
        self.in_basin_lat_lon = (
            False  # True if the lat-lon point lies within the basin's boundaries
        )
        self.in_basin_depth = np.full(
            (n_depths), False, dtype=bool
        )  # True if the depth point lies within the basin's surfaces


class PartialBasinSurfaceDepths:
    """
    Maintains depths for each basin surface at a specific lat-lon point.

    Parameters
    ----------
    basin_data : BasinData
        BasinData object referencing boundaries, surfaces, submodels.

    Attributes
    ----------
    depths : np.ndarray
        Array of depths for each basin surface.
    basin : BasinData
        BasinData object referencing boundaries, surfaces, submodels.

    """

    def __init__(self, basin_data: BasinData):
        """
        Initialize the PartialBasinSurfaceDepths object.

        """
        # self.depths[i] is the depth of the i-th surfaces
        self.depths = np.full(len(basin_data.surfaces), np.nan, dtype=np.float64)
        self.basin = basin_data

    def determine_basin_surface_depths(
        self,
        inbasin: InBasin,
        mesh_vector: MeshVector,
    ):
        """
        Interpolate surface depths for a given lat-lon.

        Parameters
        ----------
        inbasin : InBasin
            Flags if location is within the basin.
        mesh_vector : MeshVector
            Contains the lat-lon point and depths.
        """
        assert (
            inbasin.in_basin_lat_lon
        )  # this is only executed if inbasin.in_basin_lat_lon is True.

        for surface_ind, surface in enumerate(inbasin.basin_data.surfaces):
            adjacent_points = AdjacentPoints.find_basin_adjacent_points(
                surface.lats, surface.lons, mesh_vector.lat, mesh_vector.lon
            )
            # TODO: check if in_surface_bounds is True
            x1 = surface.lons[adjacent_points.lon_ind[0]]
            x2 = surface.lons[adjacent_points.lon_ind[1]]
            y1 = surface.lats[adjacent_points.lat_ind[0]]
            y2 = surface.lats[adjacent_points.lat_ind[1]]
            q11 = surface.raster[adjacent_points.lon_ind[0]][adjacent_points.lat_ind[0]]
            q12 = surface.raster[adjacent_points.lon_ind[0]][adjacent_points.lat_ind[1]]

            q21 = surface.raster[adjacent_points.lon_ind[1]][adjacent_points.lat_ind[0]]

            q22 = surface.raster[adjacent_points.lon_ind[1]][adjacent_points.lat_ind[1]]
            # TODO: refactor with
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html#scipy.interpolate.RegularGridInterpolator
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
        Ensure depths follow stratigraphy, overriding shallower depths as needed.
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

    # Superseded by determine_basin_surface_above_vectorized(), but kept for reference

    # def determine_basin_surface_above(self, depth: float):
    #     """
    #     Determine the index of the basin surface directly above the given depth.
    #
    #     Parameters
    #     ----------
    #     depth : float
    #         The depth of the grid point to determine the properties at.
    #
    #     Returns
    #     -------
    #     int
    #         Index of the surface directly above the grid point.
    #     """
    #     # self.depths is in decreasing order
    #
    #     valid_indices = np.where((~np.isnan(self.depths)) & (self.depths >= depth))[0]
    #     return valid_indices[-1] if valid_indices.size > 0 else 0  # the last one

    def determine_basin_surface_above_vectorized(self, depths: np.ndarray):
        """
        Vectorized approach for finding the indices of the basin surfaces directly above each of the given `depths`.

        Parameters
        ----------
        depths : np.ndarray
            Array of depth values to test.

        Returns
        -------
        np.ndarray
            Indices of surfaces that are directly above each depth.
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

    def enforce_basin_surface_depths(
        self,
        in_basin: InBasin,
        mesh_vector: MeshVector,
    ):
        """
        Enforce hierarchy of surfaces, then mark which depths lie within the basin.

        Parameters
        ----------
        in_basin : InBasin
            Tracks if a lat-lon is inside the basin.
        mesh_vector : MeshVector
            Coordinates for the point of interest.
    
    Raises
    ---------
    ValueError
            If the mesh vector is not contained in the basin.

        Raises
            ---------
            ValueError
                    If the mesh vector is not contained in the basin.
        """
        if not in_basin.in_basin_lat_lon:
            raise ValueError("Point is not contained in basin")

        self.enforce_surface_depths()
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
        Calculate and enforce basin surface depths for a lat-lon if in basin.

        Parameters
        ----------
        in_basin : InBasin
            Tracks whether the lat-lon is inside basin boundaries.
        mesh_vector : MeshVector
            Contains the lat-lon point and depths.
        """
        self.determine_basin_surface_depths(in_basin, mesh_vector)
        self.enforce_basin_surface_depths(in_basin, mesh_vector)


class BasinSurfaceRead:
    """
    Basic container for reading a single basin surface grid and storing
    latitude, longitude, and raster data.

    Parameters
    ----------
    nlat : int
        Number of latitudes
    nlon : int
        Number of longitudes

    Attributes
    ----------
    lats : np.ndarray
        Array of latitudes.
    lons : np.ndarray
        Array of longitudes.
    raster : np.ndarray
        2D array of raster data.
    max_lat : float
        Maximum latitude.
    min_lat : float
        Minimum latitude.

    """

    def __init__(self, nlat: int, nlon: int):
        """
        Initialize the BasinSurfaceRead object.

        """
        self.lats = np.zeros(nlat)
        self.lons = np.zeros(nlon)
        self.raster = np.zeros((nlon, nlat))
        self.max_lat = None
        self.min_lat = None
        self.max_lon = None
        self.min_lon = None
