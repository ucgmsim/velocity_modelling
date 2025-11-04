"""
Basin Model Module

This module manages basin data representation, surface interpolation, and basin membership
determination in the velocity model. It handles basin boundaries, surfaces, and submodels
with proper logging throughout the processing workflow.

"""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path

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
# Worker functions for parallel processing
# These are module-level to work with multiprocessing
# ============================================================================

_WORKER_BASIN_MEMBERSHIP = None
_WORKER_PARTIAL_MESH_LIST = None


def _init_basin_membership_worker(
    basin_data_list: list[BasinData],
    min_lons: np.ndarray,
    max_lons: np.ndarray,
    min_lats: np.ndarray,
    max_lats: np.ndarray,
    partial_global_mesh_list: list[PartialGlobalMesh],
) -> None:
    """
    Initialize worker process with shared data for parallel basin membership computation.

    Parameters
    ----------
    basin_data_list : list[BasinData]
        List of basin data objects.
    min_lons, max_lons, min_lats, max_lats : np.ndarray
        Bounding box arrays for each basin.
    partial_global_mesh_list : list[PartialGlobalMesh]
        List of partial global mesh objects for each row.
    """
    global _WORKER_BASIN_MEMBERSHIP, _WORKER_PARTIAL_MESH_LIST

    # Store data needed for basin membership checks
    _WORKER_BASIN_MEMBERSHIP = {
        "basin_data_list": basin_data_list,
        "min_lons": min_lons,
        "max_lons": max_lons,
        "min_lats": min_lats,
        "max_lats": max_lats,
    }
    _WORKER_PARTIAL_MESH_LIST = partial_global_mesh_list


def _find_containing_basins(lat: float, lon: float) -> list[int]:
    """
    Worker function to find all basins containing a point.

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
    data = _WORKER_BASIN_MEMBERSHIP

    # Vectorized bounding box check
    inside_bbox = (
        (data["min_lons"] <= lon)
        & (lon <= data["max_lons"])
        & (data["min_lats"] <= lat)
        & (lat <= data["max_lats"])
    )

    candidate_indices = np.where(inside_bbox)[0]

    # Detailed polygon check for candidates
    return [
        idx
        for idx in candidate_indices
        if determine_basin_contains_lat_lon(
            data["basin_data_list"][idx].boundaries, lat, lon
        )
    ]


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
    pgm = _WORKER_PARTIAL_MESH_LIST[j]
    nx = len(pgm.lat)
    row = []

    for k in range(nx):
        lat = pgm.lat[k]
        lon = pgm.lon[k]
        basin_indices = _find_containing_basins(lat, lon)
        row.append(basin_indices)

    return j, row


class BasinMembership:
    """
    Basin membership handler - decoupled from mesh structures.

    Provides efficient basin membership checks using bounding boxes and
    point-in-polygon tests. Optionally preprocesses smoothing boundary
    membership for velocity model smoothing at basin edges.

    This class works for both:
    - Isolated points (stations, 1D profiles, thresholds)
    - Dense grids (3D velocity models with preprocessed membership)

    Parameters
    ----------
    basin_data_list : list[BasinData]
        List of basin data objects containing boundaries.
    smooth_boundary : SmoothingBoundary, optional
        If provided, preprocesses basin membership for all smoothing points.
        This enables correct smoothing behavior in assign_qualities().
    logger : Logger, optional
        Logger instance for status reporting.

    Attributes
    ----------
    basin_data_list : list[BasinData]
        List of basin data objects.
    n_basins : int
        Number of basins.
    smoothing_boundary_basin_indices : list[list[int]] or None
        Preprocessed basin indices for each smoothing boundary point.
        None if no smoothing boundary was provided.
    grid_basin_membership : list[list[list[int]]] or None
        Preprocessed basin membership for dense grids [ny][nx] -> list[int].
        None for isolated point workflows.
    min_basin_boundary_lons : np.ndarray
        Minimum longitude for each basin (for bounding box filtering).
    max_basin_boundary_lons : np.ndarray
        Maximum longitude for each basin.
    min_basin_boundary_lats : np.ndarray
        Minimum latitude for each basin.
    max_basin_boundary_lats : np.ndarray
        Maximum latitude for each basin.

    Examples
    --------
    For isolated points (stations, 1D profiles):

    >>> basin_membership = BasinMembership(
    ...     basin_data_list,
    ...     smooth_boundary=nz_tomography_data.smooth_boundary,
    ...     logger=logger
    ... )
    >>> basin_indices = basin_membership.check_one_station(lat=-41.5, lon=174.5)

    For dense grids (3D velocity models):

    >>> basin_membership, partial_mesh_list = BasinMembership.from_dense_grid(
    ...     global_mesh,
    ...     basin_data_list,
    ...     smooth_boundary=nz_tomography_data.smooth_boundary,
    ...     np_workers=8,
    ...     logger=logger
    ... )
    >>> basin_indices = basin_membership.get_basin_membership(x=10, y=20)
    """

    def __init__(
        self,
        basin_data_list: list[BasinData],
        smooth_boundary: SmoothingBoundary | None = None,
        logger: Logger | None = None,
    ):
        """
        Initialize basin membership handler.

        This constructor is used directly for isolated point workflows.
        For dense grid workflows, use from_dense_grid() classmethod instead.
        """
        if logger is None:
            self.logger = logging.getLogger("velocity_model.basin_membership")
        else:
            self.logger = logger

        self.basin_data_list = basin_data_list
        self.n_basins = len(basin_data_list)

        # Precompute bounding boxes for efficient filtering
        self._precompute_bounding_boxes()

        # Optional: grid membership (only set by from_dense_grid())
        self.grid_basin_membership = None

        # Optional: smoothing boundary membership
        self.smoothing_boundary_basin_indices = None
        if smooth_boundary is not None and smooth_boundary.n_points > 0:
            self._preprocess_smoothing_boundary(smooth_boundary)
            self.logger.log(
                logging.INFO,
                f"Preprocessed {smooth_boundary.n_points} smoothing boundary points",
            )

        self.logger.log(
            logging.INFO, f"Initialized BasinMembership for {self.n_basins} basins"
        )

    def _precompute_bounding_boxes(self):
        """Precompute bounding boxes for all basins for efficient filtering."""
        boundary_arrays = [
            np.vstack(basin.boundaries) for basin in self.basin_data_list
        ]

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

    def _preprocess_smoothing_boundary(self, smooth_boundary: SmoothingBoundary):
        """
        Precompute basin membership for all smoothing boundary points.

        This preprocessing is done once during initialization to avoid
        repeated point-in-polygon checks during velocity assignment when
        smoothing is applied.

        Parameters
        ----------
        smooth_boundary : SmoothingBoundary
            Smoothing boundary with lat/lon coordinates.
        """
        n_points = smooth_boundary.n_points
        # Vectorized basin membership check for all smoothing points
        lats = np.array(smooth_boundary.lats)
        lons = np.array(smooth_boundary.lons)
        self.smoothing_boundary_basin_indices = self.check_stations(lats, lons)

        # Log statistics
        n_in_basin = sum(len(b) > 0 for b in self.smoothing_boundary_basin_indices)
        n_outside = n_points - n_in_basin
        self.logger.log(
            logging.DEBUG,
            f"Smoothing boundary: {n_in_basin} points in basins, "
            f"{n_outside} points outside basins",
        )

    def check_stations(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> list[list[int]]:
        """
        Check basin membership for multiple stations efficiently (vectorized).

        This method vectorizes the bounding box checks across all stations,
        making it more efficient than calling find_all_containing_basins()
        in a loop.

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

        Examples
        --------
        >>> lats = np.array([-41.5, -42.0, -43.5])
        >>> lons = np.array([174.5, 175.0, 172.5])
        >>> membership = basin_membership.check_stations(lats, lons)
        >>> print(membership)
        [[0, 1], [], [2]]  # First station in basins 0 and 1, second outside all, third in basin 2
        """
        n_stations = len(lats)
        station_basin_membership = []

        # Vectorized bounding box check for all stations at once
        # Shape: (n_stations, n_basins)
        lats_expanded = lats[:, np.newaxis]  # (n_stations, 1)
        lons_expanded = lons[:, np.newaxis]  # (n_stations, 1)

        inside_bbox = (
            (self.min_basin_boundary_lons <= lons_expanded)
            & (lons_expanded <= self.max_basin_boundary_lons)
            & (self.min_basin_boundary_lats <= lats_expanded)
            & (lats_expanded <= self.max_basin_boundary_lats)
        )  # Shape: (n_stations, n_basins)

        # For each station, check detailed polygon membership for candidate basins
        for i in range(n_stations):
            candidate_indices = np.where(inside_bbox[i])[0]

            # Detailed polygon check for candidates
            basin_indices = [
                idx
                for idx in candidate_indices
                if determine_basin_contains_lat_lon(
                    self.basin_data_list[idx].boundaries, lats[i], lons[i]
                )
            ]
            station_basin_membership.append(basin_indices)

        return station_basin_membership

    def check_one_station(self, lat: float, lon: float) -> list[int]:
        """
        Check basin membership for a single (lat, lon) point.

        This is a convenience method that wraps check_stations_in_basin()
        for a single point.

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
            Returns empty list if point is outside all basins.

        See Also
        --------
        check_stations_in_basin : More efficient for multiple points
        """
        # Convert to arrays and use vectorized method
        result = self.check_stations(np.array([lat]), np.array([lon]))
        return result[0]

    @classmethod
    def from_dense_grid(
        cls,
        global_mesh: GlobalMesh,
        basin_data_list: list[BasinData],
        smooth_boundary: SmoothingBoundary | None = None,
        logger: Logger | None = None,
        np_workers: int | None = None,
    ) -> tuple[BasinMembership, list]:
        """
        Create BasinMembership for dense grid with preprocessed membership.

        Parameters
        ----------
        global_mesh : GlobalMesh
            The global mesh for the 3D velocity model.
        basin_data_list : list[BasinData]
            List of basin data objects.
        smooth_boundary : SmoothingBoundary, optional
            Smoothing boundary to preprocess.
        logger : Logger, optional
            Logger instance.
        np_workers : int, optional
            Number of parallel workers for preprocessing.

        Returns
        -------
        tuple[BasinMembership, list[PartialGlobalMesh]]
            Basin membership handler with preprocessed grid data,
            and list of partial global meshes.
        """

        # Create instance without smoothing boundary (we'll preprocess it separately)
        membership = cls(basin_data_list, smooth_boundary=None, logger=logger)

        # Get mesh dimensions
        nx, ny = global_mesh.lat.shape

        # Create partial meshes
        partial_global_mesh_list = [
            PartialGlobalMesh(global_mesh, j) for j in range(ny)
        ]

        # Initialize grid membership structure
        membership.grid_basin_membership = [[[] for _ in range(nx)] for _ in range(ny)]

        # Determine if parallel processing should be used
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

            logger.log(
                logging.INFO,
                f"Parallelizing basin membership over {ny} rows: "
                f"workers={n_workers} ({start_method})",
            )

            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_init_basin_membership_worker,
                initargs=(
                    membership.basin_data_list,
                    membership.min_basin_boundary_lons,
                    membership.max_basin_boundary_lons,
                    membership.min_basin_boundary_lats,
                    membership.max_basin_boundary_lats,
                    partial_global_mesh_list,
                ),
            ) as ex:
                futures = [ex.submit(_compute_membership_row, j) for j in range(ny)]
                for fut in as_completed(futures):
                    j, row = fut.result()
                    membership.grid_basin_membership[j] = row

            logger.log(
                logging.INFO, f"Parallel preprocessing complete for {nx}×{ny} grid"
            )
        else:
            # Serial fallback - vectorized per row
            logger.log(
                logging.INFO,
                f"Serial preprocessing for {nx}×{ny} grid (vectorized per row)",
            )
            for j in range(ny):
                pgm = partial_global_mesh_list[j]
                # Vectorize the entire row at once
                row_membership = membership.check_stations(pgm.lat, pgm.lon)
                membership.grid_basin_membership[j] = row_membership

        # Now preprocess smoothing boundary if provided
        if smooth_boundary is not None and smooth_boundary.n_points > 0:
            logger.log(
                logging.DEBUG,
                f"Preprocessing smooth boundary with {smooth_boundary.n_points} points",
            )
            membership._preprocess_smoothing_boundary(smooth_boundary)

        logger.log(
            logging.INFO,
            f"Initialized BasinMembership for dense grid ({nx}×{ny} points) "
            f"with {len(basin_data_list)} basins",
        )

        return membership, partial_global_mesh_list

    def get_basin_membership(self, x: int, y: int) -> list[int]:
        """
        Get preprocessed basin membership for a grid point.

        This method is only available when the instance was created via
        from_dense_grid() with preprocessed grid membership.

        For isolated points, use find_all_containing_basins() instead.

        Parameters
        ----------
        x : int
            X-index in the grid.
        y : int
            Y-index in the grid.

        Returns
        -------
        list[int]
            List of basin indices for this grid point.

        Raises
        ------
        ValueError
            If this instance does not have preprocessed grid membership.
        """
        if self.grid_basin_membership is None:
            raise ValueError(
                "get_basin_membership() requires preprocessed grid data. "
                "This instance was created for isolated points. "
                "Use find_all_containing_basins(lat, lon) instead, or "
                "create instance using from_dense_grid()."
            )
        return self.grid_basin_membership[y][x]


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
