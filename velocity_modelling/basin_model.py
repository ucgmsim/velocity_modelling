"""
Basin Model Module

This module manages basin data representation, surface interpolation, and basin membership
determination in the velocity model. It handles basin boundaries, surfaces, and submodels
with proper logging throughout the processing workflow.

It provides two membership paths:
- StationBasinMembership: efficient membership checks for scattered stations.
- MeshBasinMembership (formerly InBasinGlobalMesh): precomputes membership on dense model meshes.

Notes
-----
- For isolated-station workflows (e.g., 1D profiles, threshold computations), downstream
  functions may accept mesh_basin_membership=None as long as in_basin_list is pre-populated
  with correct in_basin_lat_lon flags for each basin. This avoids unnecessary mesh-wide
  preprocessing when only a few locations are needed.
"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Self

import numpy as np
from numba import njit

from qcore import point_in_polygon
from velocity_modelling.geometry import (
    AdjacentPoints,
    GlobalMesh,
    MeshVector,
    PartialGlobalMesh,
    SmoothingBoundary,
    point_on_vertex,
)
from velocity_modelling.interpolate import (
    bi_linear_interpolation,
)
from velocity_modelling.registry import CVMRegistry


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
        self,
        cvm_registry: CVMRegistry,
        basin_name: str,
        logger: Logger | None = None,
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

        self.logger.log(logging.DEBUG, f"Basin {basin_name} fully loaded.")


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


# ============================================================================
# StationBasinMembership - For scattered point queries without mesh creation
# ============================================================================


class StationBasinMembership:
    """
    Efficient basin membership checker for scattered station points.

    This class provides fast basin membership checks for arbitrary (lat, lon) points
    without requiring mesh generation. It pre-computes bounding boxes for efficient
    filtering and uses direct point-in-polygon checks only on candidate basins.

    Use this class when:
    - Processing scattered station points (not dense grids)
    - Query count << total potential grid points
    - Smoothing boundary processing is not needed

    Parameters
    ----------
    basin_data_list : list[BasinData]
        List of BasinData objects containing basin boundaries.
    logger : Logger, optional
        Logger instance for status reporting.

    Attributes
    ----------
    basin_data_list : list[BasinData]
        List of basin data objects.
    n_basins : int
        Number of basins.
    min_basin_boundary_lons : np.ndarray
        Minimum longitude for each basin boundary.
    max_basin_boundary_lons : np.ndarray
        Maximum longitude for each basin boundary.
    min_basin_boundary_lats : np.ndarray
        Minimum latitude for each basin boundary.
    max_basin_boundary_lats : np.ndarray
        Maximum latitude for each basin boundary.
    logger : Logger
        Logger instance.

    Examples
    --------
    >>> checker = StationBasinMembership(basin_data_list, logger)
    >>> basin_indices = checker.check_point_in_basin(lat=-41.5, lon=174.5)
    >>> print(f"Point is in basins: {basin_indices}")
    """

    def __init__(
        self,
        basin_data_list: list[BasinData],
        logger: Logger | None = None,
    ):
        """Initialize the StationBasinMembership checker."""
        if logger is None:
            self.logger = logging.getLogger("velocity_model.station_basin_membership")
        else:
            self.logger = logger

        self.basin_data_list = basin_data_list
        self.n_basins = len(basin_data_list)

        # Pre-compute bounding boxes for all basins
        self._precompute_bounding_boxes()

        self.logger.log(
            logging.INFO,
            f"Initialized station basin membership checker for {self.n_basins} basins",
        )

    def _precompute_bounding_boxes(self):
        """Pre-compute bounding boxes for efficient filtering."""
        boundary_arrays = [
            np.vstack(basin.boundaries)  # Merge all boundary arrays for each basin
            for basin in self.basin_data_list
        ]

        # Compute min/max lat/lon per basin
        self.min_basin_boundary_lons = np.array(
            [np.min(boundary[:, 0]) for boundary in boundary_arrays]
        )
        self.max_basin_boundary_lons = np.array(
            [np.max(boundary[:, 0]) for boundary in boundary_arrays]
        )
        self.min_basin_boundary_lats = np.array(
            [np.min(boundary[:, 1]) for boundary in boundary_arrays]
        )
        self.max_basin_boundary_lats = np.array(
            [np.max(boundary[:, 1]) for boundary in boundary_arrays]
        )

    def check_point_in_basin(self, lat: float, lon: float) -> list[int]:
        """
        Check which basins contain a given (lat, lon) point.

        This method first filters basins using bounding boxes, then performs
        detailed polygon containment checks only on candidate basins.

        Parameters
        ----------
        lat : float
            Latitude of the point.
        lon : float
            Longitude of the point.

        Returns
        -------
        list[int]
            List of basin indices that contain the point.
        """
        # Step 1: Vectorized bounding box check
        inside_bbox = (
            (self.min_basin_boundary_lons <= lon)
            & (lon <= self.max_basin_boundary_lons)
            & (self.min_basin_boundary_lats <= lat)
            & (lat <= self.max_basin_boundary_lats)
        )

        # Get candidate basin indices
        candidate_indices = np.where(inside_bbox)[0]

        # Step 2: Detailed polygon check (only for candidates)
        return [
            idx
            for idx in candidate_indices
            if determine_basin_contains_lat_lon(
                self.basin_data_list[idx].boundaries, lat, lon
            )
        ]

    def check_stations_in_basin(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> list[list[int]]:
        """
        Check basin membership for multiple stations efficiently.

        Parameters
        ----------
        lats : np.ndarray
            Array of station latitudes.
        lons : np.ndarray
            Array of station longitudes.

        Returns
        -------
        list[list[int]]
            List of basin indices for each station. Each element is a list
            of basin indices that contain that station.
        """
        n_stations = len(lats)
        station_basin_membership = []

        for i in range(n_stations):
            basin_indices = self.check_point_in_basin(lats[i], lons[i])
            station_basin_membership.append(basin_indices)

        return station_basin_membership

    def is_point_in_any_basin(self, lat: float, lon: float) -> bool:
        """
        Check if a point is in ANY basin (faster than getting all basins).

        Parameters
        ----------
        lat : float
            Latitude of the point.
        lon : float
            Longitude of the point.

        Returns
        -------
        bool
            True if point is in at least one basin, False otherwise.
        """
        basin_indices = self.check_point_in_basin(lat, lon)
        return len(basin_indices) > 0


def compute_sigma_for_stations(
    station_basin_membership: list[list[int]],
    in_basin_sigma: float = 0.3,
    out_basin_sigma: float = 0.5,
) -> np.ndarray:
    """
    Compute sigma values for stations based on basin membership.

    This function assigns uncertainty (sigma) values to stations based on whether
    they are inside or outside basin boundaries. Stations within basins typically
    have lower uncertainty due to better constrained velocity models.

    The sigma values represent the standard deviation of the depth-to-velocity
    threshold (e.g., Z1.0, Z2.5) and are used in ground motion modeling to
    quantify uncertainty in basin depth estimates.

    Parameters
    ----------
    station_basin_membership : list[list[int]]
        List of basin indices for each station. Each element is a list of
        basin indices that contain that station. Empty list means station
        is outside all basins.
    in_basin_sigma : float, optional
        Sigma value for stations inside basins (default: 0.3).
        Lower value reflects higher confidence in basin depth estimates.
    out_basin_sigma : float, optional
        Sigma value for stations outside basins (default: 0.5).
        Higher value reflects greater uncertainty outside basins.

    Returns
    -------
    np.ndarray
        Array of sigma values (float64) for each station.

    Examples
    --------
    >>> # 3 stations: first in basins 0 and 1, second outside all basins, third in basin 2
    >>> membership = [[0, 1], [], [2]]
    >>> sigmas = compute_sigma_for_stations(membership)
    >>> print(sigmas)
    [0.3 0.5 0.3]

    >>> # Custom sigma values
    >>> sigmas = compute_sigma_for_stations(membership, 0.2, 0.6)
    >>> print(sigmas)
    [0.2 0.6 0.2]

    >>> # All stations in basins
    >>> membership = [[0], [1], [0, 1]]
    >>> sigmas = compute_sigma_for_stations(membership)
    >>> print(sigmas)
    [0.3 0.3 0.3]

    Notes
    -----
    The default sigma values (0.3 for in-basin, 0.5 for out-of-basin) are based
    on empirical observations and match the values used in the original C
    implementation (get_z.py).

    A station is considered "in basin" if it's inside at least one basin boundary.
    Stations can be in multiple basins simultaneously.
    """
    n_stations = len(station_basin_membership)
    sigma_values = np.zeros(n_stations, dtype=np.float64)

    for i in range(n_stations):
        # If station is in at least one basin
        if len(station_basin_membership[i]) > 0:
            sigma_values[i] = in_basin_sigma
        else:
            sigma_values[i] = out_basin_sigma

    return sigma_values


# ============================================================================
# MeshBasinMembership - For dense grid queries with preprocessing
# ============================================================================


_WORK_INB = None
_WORK_PGM_LIST = None


def _init_inbasin_worker(
    mesh_basin_membership: MeshBasinMembership,
    partial_global_mesh_list: list[PartialGlobalMesh],
) -> None:
    """
    Initialize worker process with shared data for parallel basin membership computation.

    Parameters
    ----------
    mesh_basin_membership : MeshBasinMembership
        The basin mesh object containing basin data and boundaries.
    partial_global_mesh_list : list[PartialGlobalMesh]
        List of partial global mesh objects for each row.
    """
    # Called once in each worker
    global _WORK_INB, _WORK_PGM_LIST
    _WORK_INB = mesh_basin_membership
    _WORK_PGM_LIST = partial_global_mesh_list


def _compute_membership_row(j: int) -> tuple[int, list[list[int]]]:
    """
    Compute basin membership for one row j.

    Parameters
    ----------
    j : int
        Row index to compute basin membership for.

    Returns
    -------
    tuple[int, list[list[int]]]
        Tuple containing (j, row_list), where row_list is a list of lists
        of basin indices with length nx.
    """
    mesh_basin_membership = _WORK_INB
    pgm = _WORK_PGM_LIST[j]
    nx = mesh_basin_membership.nx
    row = [[] for _ in range(nx)]
    # identical to your serial inner loop
    for k in range(nx):
        lat = pgm.lat[k]
        lon = pgm.lon[k]
        row[k] = mesh_basin_membership.find_all_containing_basins(lat, lon)
    return j, row


class MeshBasinMembership:
    """
    Precomputed basin membership for every (x, y) point in a dense global mesh.

    This class is optimized for dense grid workflows (e.g., 3D velocity model generation).
    It preprocesses membership across all grid points to enable O(1) lookups during model
    generation. For scattered points, use StationBasinMembership instead.

    For isolated-station processing (1D profiles, threshold points), callers may pass
    mesh_basin_membership=None to downstream functions (e.g., assign_qualities) provided that
    in_basin_list has been pre-populated with correct in_basin_lat_lon flags.

    Parameters
    ----------
    global_mesh : GlobalMesh
        The global mesh containing lat/lon values.
    basin_data_list : list[BasinData]
        List of BasinData objects for basin membership.
    logger : Logger, optional
        Logger instance.

    Attributes
    ----------
    nx : int
        Number of x-coordinates in the global mesh.
    ny : int
        Number of y-coordinates in the global mesh.
    nz : int
        Number of z-coordinates in the global mesh.
    basin_data_list : list[BasinData]
        List of BasinData objects for basin membership.
    basin_membership : list[list[list[int]]]
        Basin indices for each (y, x) point in the mesh. Access with [y][x].
    smoothing_boundary_basin_indices : list[list[int]] | None
        For each smoothing-boundary point (in order), the list of basin indices
        that contain that boundary point. None if no smoothing boundary provided.
    min_basin_boundary_lons : np.ndarray
        Minimum longitude for each basin.
    max_basin_boundary_lons : np.ndarray
        Maximum longitude for each basin.
    min_basin_boundary_lats : np.ndarray
        Minimum latitude for each basin.
    max_basin_boundary_lats : np.ndarray
        Maximum latitude for each basin.
    logger : Logger
        Logger instance.
    """

    def __init__(
        self,
        global_mesh: GlobalMesh,
        basin_data_list: list[BasinData],
        logger: Logger | None = None,
    ):
        """
        Private constructor. Use preprocess_basin_membership() instead to create instances.
        """
        if logger is None:
            self.logger = Logger(name="velocity_model.mesh_basin_membership")
        else:
            self.logger = logger

        self.nx, self.ny = global_mesh.lat.shape
        self.nz = len(global_mesh.z)

        self.basin_data_list = basin_data_list
        self.smoothing_boundary_basin_indices = None
        self.basin_membership = None  # Will be set by preprocess_basin_membership

        # Bounding boxes per basin (filled in preprocess_basin_membership)
        self.min_basin_boundary_lons = self.max_basin_boundary_lons = (
            self.min_basin_boundary_lats
        ) = self.max_basin_boundary_lats = None

    @classmethod
    def preprocess_basin_membership(
        cls,
        global_mesh: GlobalMesh,
        basin_data_list: list[BasinData],
        logger: Logger | None = None,
        smooth_bound: SmoothingBoundary | None = None,
        np_workers: int | None = None,
    ) -> tuple[Self, list[PartialGlobalMesh]]:
        """
        Preprocess basin membership for a given global mesh to speed up the velocity model generation
        This method is the recommended way to create an MeshBasinMembership object.

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
        np_workers : int, optional
            Number of workers for parallel processing (default is None).

        Returns
        -------
        tuple of (MeshBasinMembership, list of PartialGlobalMesh)
            Mesh membership object and list of partial slices.
        """
        if logger is None:
            logger = Logger(name="velocity_model.mesh_basin_membership")

        mesh_basin_membership = cls(global_mesh, basin_data_list, logger)

        # Use object dtype to store lists of basin indices
        # Initialize basin_membership as an (ny, nx) array of empty lists
        mesh_basin_membership.basin_membership = [
            [[] for _ in range(mesh_basin_membership.nx)]
            for _ in range(mesh_basin_membership.ny)
        ]

        boundary_arrays = [
            np.vstack(basin.boundaries)  # Merge all boundary arrays for each basin
            for basin in basin_data_list
        ]

        # Compute min/max lat/lon per basin
        mesh_basin_membership.min_basin_boundary_lons = np.array(
            [np.min(boundary[:, 0]) for boundary in boundary_arrays]
        )
        mesh_basin_membership.max_basin_boundary_lons = np.array(
            [np.max(boundary[:, 0]) for boundary in boundary_arrays]
        )
        mesh_basin_membership.min_basin_boundary_lats = np.array(
            [np.min(boundary[:, 1]) for boundary in boundary_arrays]
        )
        mesh_basin_membership.max_basin_boundary_lats = np.array(
            [np.max(boundary[:, 1]) for boundary in boundary_arrays]
        )

        if smooth_bound is not None:
            logger.log(
                logging.DEBUG,
                f"Initializing smooth boundary with {smooth_bound.n_points} points",
            )
            mesh_basin_membership.preprocess_smooth_bound(smooth_bound)
            logger.log(
                logging.DEBUG,
                f"Pre-processed smooth boundary membership for {smooth_bound.n_points} points.",
            )
            logger.log(
                logging.DEBUG,
                f"mesh_basin_membership.smoothing_boundary_basin_indices after preprocess: "
                f"{mesh_basin_membership.smoothing_boundary_basin_indices}",
            )
        else:
            logger.log(logging.DEBUG, "smooth_bound is None")

        nx, ny = mesh_basin_membership.nx, mesh_basin_membership.ny
        partial_global_mesh_list = [
            PartialGlobalMesh(global_mesh, j) for j in range(ny)
        ]

        is_parallel = (np_workers or 1) > 1 and ny > 1

        if is_parallel:
            import multiprocessing as mp
            import os
            from concurrent.futures import ProcessPoolExecutor, as_completed

            n_workers = min(np_workers or (os.cpu_count() or 1), ny)

            # Prefer 'fork' on Linux to inherit big, read-only objects without pickling
            try:
                ctx = mp.get_context("fork")
                start_method = "fork"
            except ValueError:
                # Fallback: spawn (safe everywhere, but will pickle more)
                ctx = mp.get_context("spawn")
                start_method = "spawn"

            if logger:
                logger.log(
                    logging.INFO,
                    f"Parallelizing basin membership over rows: workers={n_workers} ({start_method}).",
                )

            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_init_inbasin_worker,
                initargs=(mesh_basin_membership, partial_global_mesh_list),
            ) as ex:
                futures = [ex.submit(_compute_membership_row, j) for j in range(ny)]
                for fut in as_completed(futures):
                    j, row = fut.result()
                    mesh_basin_membership.basin_membership[j] = row
        else:
            # --- Serial fallback (default) ---
            for j in range(ny):
                pgm = partial_global_mesh_list[j]
                for k in range(nx):
                    lat = pgm.lat[k]
                    lon = pgm.lon[k]
                    mesh_basin_membership.basin_membership[j][k] = (
                        mesh_basin_membership.find_all_containing_basins(lat, lon)
                    )

        logger.log(
            logging.INFO,
            f"Pre-processed basin membership for {len(basin_data_list)} basins.",
        )
        return (mesh_basin_membership, partial_global_mesh_list)

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

        Raises
        ------
        ValueError
            If basin membership has not been preprocessed.
        """
        # Correct guard: ensure membership was computed
        if self.basin_membership is None:
            raise ValueError("Basin membership not pre-processed.")
        return self.basin_membership[y][x]

    def find_all_containing_basins(self, lat: float, lon: float) -> list[int]:
        """
        Determine all basins that contain a given (lat, lon) without using precomputed membership.

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
        self.smoothing_boundary_basin_indices = [[] for _ in range(n_points)]

        for i in range(n_points):
            lat = smooth_bound.lats[i]
            lon = smooth_bound.lons[i]
            self.smoothing_boundary_basin_indices[i] = self.find_all_containing_basins(
                lat, lon
            )

        self.logger.log(
            logging.DEBUG,
            "smoothing_boundary_basin_indices initialized with length "
            f"{len(self.smoothing_boundary_basin_indices)}",
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
        Array of depths for each basin surface. depths[i] is the depth of the i-th surfaces
    basin : BasinData
        BasinData object referencing boundaries, surfaces, submodels.

    """

    def __init__(self, basin_data: BasinData):
        """
        Initialize the PartialBasinSurfaceDepths object.

        """
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

        for surface_ind, surface in enumerate(inbasin.basin_data.surfaces):
            adjacent_points = AdjacentPoints.find_basin_adjacent_points(
                surface.lats, surface.lons, mesh_vector.lat, mesh_vector.lon
            )
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

    def determine_basin_surface_above_vectorized(self, depths: np.ndarray):
        """
        Find the indices of the basin surfaces directly above each of the given `depths`.

        Non-vectorized version simply returns the last index of the valid depths that are greater than or equal to
        the given 'depth'.

        ```
        def determine_basin_surface_above(self, depth: float):
            valid_indices = np.where((~np.isnan(self.depths)) & (self.depths >= depth))[0]
            return valid_indices[-1] if valid_indices.size > 0 else 0  # the last one, self.depths is in decreasing order
        ```

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

        """
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

        Raises
        ------
        ValueError
            If the mesh vector is not contained in the basin.
        """
        if not in_basin.in_basin_lat_lon:
            raise ValueError("Point is not contained in basin")

        self.determine_basin_surface_depths(in_basin, mesh_vector)
        self.enforce_basin_surface_depths(in_basin, mesh_vector)


class BasinSurfaceRead:
    """
    Basic container for reading a single basin surface grid and storing
    latitude, longitude, and raster data.

    Parameters
    ----------
    file_path : Path
        Path to the basin surface file.
    lats : np.ndarray
        Array of latitudes.
    lons : np.ndarray
        Array of longitudes.
    raster : np.ndarray
        2D array of raster data.


    Attributes
    ----------
    file_path : Path
        Path to the basin surface file.
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

    def __init__(
        self, file_path: Path, lats: np.ndarray, lons: np.ndarray, raster: np.ndarray
    ):
        """
        Initialize the BasinSurfaceRead object.

        """
        self.file_path = file_path
        self.lats = lats
        self.lons = lons
        self.raster = raster
        self.max_lat = max(self.lats[0], self.lats[-1])
        self.min_lat = min(self.lats[0], self.lats[-1])
        self.max_lon = max(self.lons[0], self.lons[-1])
        self.min_lon = min(self.lons[0], self.lons[-1])
