import logging
import sys
from logging import Logger
from typing import Dict, List

import numpy as np
from numba import njit

from velocity_modelling.cvm.interpolate import (
    bi_linear_interpolation,
    linear_interpolation,
)
from velocity_modelling.cvm.geometry import (
    AdjacentPoints,
    MeshVector,
)
from velocity_modelling.cvm.velocity1d import VelocityModel1D


class GlobalSurfaceRead:
    def __init__(
        self, latitudes: np.ndarray, longitudes: np.ndarray, raster: np.ndarray
    ):
        """
        Initialize the GlobalSurfaceRead.

        Parameters
        ----------
        latitudes : np.ndarray
            The latitude values.
        longitudes : np.ndarray
            The longitude values.
        raster : np.ndarray
            The raster values.

        """
        self.lati = latitudes
        self.loni = longitudes
        self.raster = raster
        self.max_lat = None
        self.min_lat = None
        self.max_lon = None
        self.min_lon = None

        self.max_lat = max(self.lati)
        self.min_lat = min(self.lati)
        self.max_lon = max(self.loni)
        self.max_lat = min(self.loni)

    @property
    def nlat(self):
        return len(self.lati)

    @property
    def nlon(self):
        return len(self.loni)


class GlobalSurfaces:
    def __init__(self, surfaces: List[GlobalSurfaceRead]):
        """
        Initialize the GlobalSurfaces.

        Parameters
        ----------
        surfaces : List[GlobalSurfaceRead]
            List of GlobalSurfaceRead objects.

        """
        self.surfaces = surfaces


class PartialGlobalSurfaceDepths:
    def __init__(self, n_surfaces: int):
        """
        Initialize the PartialGlobalSurfaceDepth.

        Parameters
        ----------
        n_surfaces : int
            The number of global surfaces.
        """
        self.depths = np.zeros(n_surfaces, dtype=np.float64)

    def find_global_submodel_ind(
        self,
        depth: np.float64,
    ):
        """
        Find the index of the global sub-velocity model at the given depth.

        Parameters
        ----------
        depth : float
            The depth (in m) to find the sub-velocity model index at.

        Returns
        -------
        int
            The index of the global sub-velocity model.
        """
        try:
            n_velo_ind = np.where(self.depths >= depth)[0][-1]
            if n_velo_ind == len(self.depths):
                raise ValueError("Error: depth not found in global sub-velocity model.")
        except IndexError:
            raise ValueError("Error: depth not found in global sub-velocity model.")

        return n_velo_ind

    def find_global_submodel_ind_vectorized(self, depths: np.ndarray) -> np.ndarray:
        """
        Find the indices of the global sub-velocity model for an array of depths.

        Parameters
        ----------
        depths : np.ndarray
            Array of depths (in m) to find the sub-velocity model indices for.

        Returns
        -------
        np.ndarray
            Array of indices of the global sub-velocity models.

        Raises
        ------
        ValueError
            If any depth is not found in the global sub-velocity model.
        """
        # Ensure depths is a NumPy array
        depths = np.asarray(depths)

        # Initialize output array with invalid indices
        n_velo_indices = np.full_like(depths, -1, dtype=int)

        # Valid depths must be less than or equal to the shallowest depth in self.depths
        max_depth = self.depths[0]  # Assuming self.depths is in decreasing order
        assert np.all(depths <= max_depth)
        valid_mask = depths <= max_depth

        if not np.all(valid_mask):
            invalid_depths = depths[~valid_mask]
            raise ValueError(
                f"Error: Some depths not found in global sub-velocity model: {invalid_depths}"
            )

        # Vectorized search for indices
        # self.depths is in decreasing order, so reverse for searchsorted
        indices = np.searchsorted(self.depths[::-1], depths, side="right")
        # Convert to indices in the original array
        n_velo_indices = len(self.depths) - indices - 1

        # Check for invalid indices (shouldn't happen due to valid_mask, but for safety)
        if np.any(n_velo_indices >= len(self.depths)):
            invalid_indices = np.where(n_velo_indices >= len(self.depths))[0]
            invalid_depths = depths[invalid_indices]
            raise ValueError(
                f"Error: Some depths not found in global sub-velocity model: {invalid_depths}"
            )

        return n_velo_indices

    def interpolate_global_surface_depths(
        self,
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
            Object containing calculation data and write directory.
        """
        for i in range(len(global_surfaces.surfaces)):
            global_surf_read = global_surfaces.surfaces[i]
            adjacent_points = AdjacentPoints.find_global_adjacent_points(
                global_surf_read.lati,
                global_surf_read.loni,
                mesh_vector.lat,
                mesh_vector.lon,
            )
            # self.depths[i] = interpolate_global_surface(
            #     global_surf_read, mesh_vector, adjacent_points
            # )
            self.depths[i] = interpolate_global_surface_numba(
                global_surf_read.lati,
                global_surf_read.loni,
                global_surf_read.raster,
                mesh_vector.lat,
                mesh_vector.lon,
                adjacent_points.lat_ind,
                adjacent_points.lon_ind,
                adjacent_points.in_surface_bounds,
                adjacent_points.in_lat_extension_zone,
                adjacent_points.in_lon_extension_zone,
                adjacent_points.in_corner_zone,
                adjacent_points.lat_edge_ind,
                adjacent_points.lon_edge_ind,
                adjacent_points.corner_lat_ind,
                adjacent_points.corner_lon_ind,
            )

        # Find indices where top_val < bot_val
        mask = self.depths[:-1] < self.depths[1:]

        # Apply the condition using NumPy indexing
        self.depths[1:][mask] = self.depths[:-1][mask]


@njit
def interpolate_global_surface_numba(
    lati: np.ndarray,
    loni: np.ndarray,
    raster: np.ndarray,
    lat: float,
    lon: float,
    lat_ind: np.ndarray,
    lon_ind: np.ndarray,
    in_surface_bounds: bool,
    in_lat_extension_zone: bool,
    in_lon_extension_zone: bool,
    in_corner_zone: bool,
    lat_edge_ind: int,
    lon_edge_ind: int,
    corner_lat_ind: int,
    corner_lon_ind: int,
) -> float:
    if in_surface_bounds:
        x1 = loni[lon_ind[0]]
        x2 = loni[lon_ind[1]]
        y1 = lati[lat_ind[0]]
        y2 = lati[lat_ind[1]]

        q11 = raster[lon_ind[0], lat_ind[0]]
        q12 = raster[lon_ind[0], lat_ind[1]]
        q21 = raster[lon_ind[1], lat_ind[0]]
        q22 = raster[lon_ind[1], lat_ind[1]]

        # Assume bi_linear_interpolation is also Numba-ified
        return bi_linear_interpolation(x1, x2, y1, y2, q11, q12, q21, q22, lon, lat)

    elif in_lat_extension_zone:
        p1 = loni[lon_ind[0]]
        p2 = loni[lon_ind[1]]
        v1 = raster[lon_ind[0], lat_edge_ind]
        v2 = raster[lon_ind[1], lat_edge_ind]
        return linear_interpolation(p1, p2, v1, v2, lon)

    elif in_lon_extension_zone:
        p1 = lati[lat_ind[0]]
        p2 = lati[lat_ind[1]]
        v1 = raster[lon_edge_ind, lat_ind[0]]
        v2 = raster[lon_edge_ind, lat_ind[1]]
        return linear_interpolation(p1, p2, v1, v2, lat)

    elif in_corner_zone:
        return raster[corner_lon_ind, corner_lat_ind]

    raise ValueError("Calculation of Global surface value failed.")


class TomographyData:
    def __init__(
        self,
        name: str,
        surf_depth: List[float],
        special_offshore_tapering: bool,
        vs30: GlobalSurfaceRead,
        surfaces: List[Dict[str, GlobalSurfaceRead]],
        offshore_distance_surface: GlobalSurfaceRead,
        offshore_basin_model_1d: VelocityModel1D,
        logger: Logger = None,
    ):
        """
        Initialize the TomographyData.

        Parameters
        ----------
        name : str
            The name of the tomography data.
        surf_depth : List[float]
            The list of surfaces depths.
        special_offshore_tapering : bool
            Flag for special offshore tapering.
        vs30 : GlobalSurfaceRead
            The vs30 surfaces data.
        surfaces : List[Dict[str, GlobalSurfaceRead]],
            List of surfaces data for each velocity type.
        offshore_distance_surface : GlobalSurfaceRead
            The offshore distance surfaces data.
        offshore_basin_model_1d : VelocityModel1D
            The offshore 1D model data.
        logger : Logger, optional
            Logger instance for logging messages.
        """

        self.name = name
        self.surf_depth = surf_depth
        self.surfaces = surfaces
        self.tomography_loaded = False
        self.special_offshore_tapering = special_offshore_tapering
        self.smooth_boundary = None
        self.vs30 = vs30
        self.offshore_distance_surface = offshore_distance_surface
        self.offshore_basin_model_1d = offshore_basin_model_1d
        self.tomography_loaded = True
        self.logger = logger

    def log(self, message, level=logging.INFO):
        if self.logger is not None:
            self.logger.log(level, message)
        else:
            print(message, file=sys.stderr)

    def calculate_vs30_from_tomo_vs30_surface(self, mesh_vector: MeshVector):
        """
        Calculate the Vs30 value at a given mesh vector.
        Parameters
        ----------
        mesh_vector: MeshVector
            The mesh vector containing latitude and longitude.


        Returns
        -------
        None: The result is stored in the mesh_vector's vs30 attribute.
        """
        from velocity_modelling.cvm.global_model import interpolate_global_surface_numba

        adjacent_points = AdjacentPoints.find_global_adjacent_points(
            self.vs30.lati, self.vs30.loni, mesh_vector.lat, mesh_vector.lon
        )

        # mesh_vector.vs30 = interpolate_global_surface(
        #     self.vs30, mesh_vector, adjacent_points
        # )
        mesh_vector.vs30 = interpolate_global_surface_numba(
            self.vs30.lati,
            self.vs30.loni,
            self.vs30.raster,
            mesh_vector.lat,
            mesh_vector.lon,
            adjacent_points.lat_ind,
            adjacent_points.lon_ind,
            adjacent_points.in_surface_bounds,
            adjacent_points.in_lat_extension_zone,
            adjacent_points.in_lon_extension_zone,
            adjacent_points.in_corner_zone,
            adjacent_points.lat_edge_ind,
            adjacent_points.lon_edge_ind,
            adjacent_points.corner_lat_ind,
            adjacent_points.corner_lon_ind,
        )

    def calculate_distance_from_shoreline(self, mesh_vector: MeshVector):
        """
        Calculate the distance from the shoreline for a given mesh vector.

        Parameters
        ----------
        mesh_vector : MeshVector
            The mesh vector containing latitude and longitude.

        Returns
        -------
        None : The result is stored in the mesh_vector's distance_from_shoreline attribute.
        """
        from velocity_modelling.cvm.global_model import interpolate_global_surface_numba

        # Find the adjacent points for interpolation
        adjacent_points = AdjacentPoints.find_global_adjacent_points(
            self.offshore_distance_surface.lati,
            self.offshore_distance_surface.loni,
            mesh_vector.lat,
            mesh_vector.lon,
        )

        # Interpolate the global surfaces value at the given latitude and longitude
        # mesh_vector.distance_from_shoreline = interpolate_global_surface(
        #     self.offshore_distance_surface, mesh_vector, adjacent_points
        # )
        # Extract data once from mesh_vector and adjacent_points (reused across calls)

        mesh_vector.distance_from_shoreline = interpolate_global_surface_numba(
            self.offshore_distance_surface.lati,
            self.offshore_distance_surface.loni,
            self.offshore_distance_surface.raster,
            mesh_vector.lat,
            mesh_vector.lon,
            adjacent_points.lat_ind,
            adjacent_points.lon_ind,
            adjacent_points.in_surface_bounds,
            adjacent_points.in_lat_extension_zone,
            adjacent_points.in_lon_extension_zone,
            adjacent_points.in_corner_zone,
            adjacent_points.lat_edge_ind,
            adjacent_points.lon_edge_ind,
            adjacent_points.corner_lat_ind,
            adjacent_points.corner_lon_ind,
        )
