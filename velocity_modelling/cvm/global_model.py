"""
This module provides functionality for handling global surfaces and tomography data,
including interpolation and depth calculations.

.. module:: global_model
"""

from typing import Dict, List

import numpy as np
from numba import njit

from velocity_modelling.cvm.geometry import (
    AdjacentPoints,
    MeshVector,
)
from velocity_modelling.cvm.interpolate import (
    bi_linear_interpolation,
    linear_interpolation,
)
from velocity_modelling.cvm.velocity1d import (
    VelocityModel1D,
)


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
        self.max_lat = max(self.lati)
        self.min_lat = min(self.lati)
        self.max_lon = max(self.loni)
        self.min_lon = min(self.loni)

    @property
    def nlat(self) -> int:
        """
        Get the number of latitude points.

        Returns
        -------
        int
            Number of latitude points.
        """
        return len(self.lati)

    @property
    def nlon(self) -> int:
        """
        Get the number of longitude points.

        Returns
        -------
        int
            Number of longitude points.
        """
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
        Initialize the PartialGlobalSurfaceDepths.

        Parameters
        ----------
        n_surfaces : int
            The number of global surfaces.
        """
        self.depths = np.zeros(n_surfaces, dtype=np.float64)

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
        depths = np.asarray(depths)
        max_depth = self.depths[0]
        valid_mask = depths <= max_depth

        if not np.all(valid_mask):
            invalid_depths = depths[~valid_mask]
            raise ValueError(
                f"Error: Some depths not found in global sub-velocity model: {invalid_depths}"
            )

        indices = np.searchsorted(self.depths[::-1], depths, side="right")
        n_velo_indices = len(self.depths) - indices - 1

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
    ) -> None:
        """
        Interpolate the surface depths at the lat lon location given in mesh_vector.

        Parameters
        ----------
        global_surfaces : GlobalSurfaces
            Object containing pointers to global surfaces.
        mesh_vector : MeshVector
            Object containing a single lat lon point with one or more depths.

        Returns
        -------
        None
        """
        for i in range(len(global_surfaces.surfaces)):
            global_surf_read = global_surfaces.surfaces[i]
            adjacent_points = AdjacentPoints.find_global_adjacent_points(
                global_surf_read.lati,
                global_surf_read.loni,
                mesh_vector.lat,
                mesh_vector.lon,
            )

            self.depths[i] = interpolate_global_surface(
                global_surf_read, mesh_vector.lat, mesh_vector.lon, adjacent_points
            )

        mask = self.depths[:-1] < self.depths[1:]
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
    """
    Perform specialized interpolation for global surface values.

    Parameters
    ----------
    lati : np.ndarray
        Array of latitudes for the raster.
    loni : np.ndarray
        Array of longitudes for the raster.
    raster : np.ndarray
        2D array of raster values.
    lat, lon : float
        Coordinates for interpolation.
    lat_ind, lon_ind : np.ndarray
        Indices for bounding points.
    in_surface_bounds, in_lat_extension_zone, in_lon_extension_zone, in_corner_zone : bool
        Flags indicating interpolation zone.
    lat_edge_ind, lon_edge_ind, corner_lat_ind, corner_lon_ind : int
        Edge/corner indices for fallback interpolation.

    Returns
    -------
    float
        Interpolated raster value.
    """

    if in_surface_bounds:
        x1 = loni[lon_ind[0]]
        x2 = loni[lon_ind[1]]
        y1 = lati[lat_ind[0]]
        y2 = lati[lat_ind[1]]

        q11 = raster[lon_ind[0], lat_ind[0]]
        q12 = raster[lon_ind[0], lat_ind[1]]
        q21 = raster[lon_ind[1], lat_ind[0]]
        q22 = raster[lon_ind[1], lat_ind[1]]

        # explicitly cast to float to prevent numba from regarding them as ndarray
        return bi_linear_interpolation(
            float(x1),
            float(x2),
            float(y1),
            float(y2),
            float(q11),
            float(q12),
            float(q21),
            float(q22),
            lon,
            lat,
        )

    elif in_lat_extension_zone:
        p1 = loni[lon_ind[0]]
        p2 = loni[lon_ind[1]]
        v1 = raster[lon_ind[0], lat_edge_ind]
        v2 = raster[lon_ind[1], lat_edge_ind]
        # explicitly cast to float to prevent numba from regarding them as ndarray
        return linear_interpolation(float(p1), float(p2), float(v1), float(v2), lon)

    elif in_lon_extension_zone:
        p1 = lati[lat_ind[0]]
        p2 = lati[lat_ind[1]]
        v1 = raster[lon_edge_ind, lat_ind[0]]
        v2 = raster[lon_edge_ind, lat_ind[1]]
        # explicitly cast to float to prevent numba from regarding them as ndarray
        return linear_interpolation(float(p1), float(p2), float(v1), float(v2), lat)

    elif in_corner_zone:
        return float(raster[corner_lon_ind, corner_lat_ind])

    raise ValueError("Calculation of Global surface value failed.")


def interpolate_global_surface(
    surface: GlobalSurfaceRead, lat: float, lon: float, adjacent_points: AdjacentPoints
) -> float:
    """
    Wrapper to call numba-based interpolation on a global surface.

    Parameters
    ----------
    surface : GlobalSurfaceRead
        Contains lat/lon arrays and a 2D raster.
    lat : float
        Latitude of the target point.
    lon : float
        Longitude of the target point.
    adjacent_points : AdjacentPoints
        Object containing bounding indices and zone checks.

    Returns
    -------
    float
        Interpolated surface value.
    """
    lat_ind = np.array(adjacent_points.lat_ind, dtype=np.int64)
    lon_ind = np.array(adjacent_points.lon_ind, dtype=np.int64)

    return interpolate_global_surface_numba(
        surface.lati,
        surface.loni,
        surface.raster,
        lat,
        lon,
        lat_ind,
        lon_ind,
        adjacent_points.in_surface_bounds,
        adjacent_points.in_lat_extension_zone,
        adjacent_points.in_lon_extension_zone,
        adjacent_points.in_corner_zone,
        adjacent_points.lat_edge_ind,
        adjacent_points.lon_edge_ind,
        adjacent_points.corner_lat_ind,
        adjacent_points.corner_lon_ind,
    )


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
        surfaces : List[Dict[str, GlobalSurfaceRead]]
            List of surfaces data for each velocity type.
        offshore_distance_surface : GlobalSurfaceRead
            The offshore distance surfaces data.
        offshore_basin_model_1d : VelocityModel1D
            The offshore 1D model data.
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

    def calculate_vs30_from_tomo_vs30_surface(self, mesh_vector: MeshVector) -> None:
        """
        Calculate the Vs30 value at a given mesh vector.

        Parameters
        ----------
        mesh_vector : MeshVector
            The mesh vector containing latitude and longitude.

        Returns
        -------
        None
        """
        adjacent_points = AdjacentPoints.find_global_adjacent_points(
            self.vs30.lati, self.vs30.loni, mesh_vector.lat, mesh_vector.lon
        )

        mesh_vector.vs30 = interpolate_global_surface(
            self.vs30, mesh_vector.lat, mesh_vector.lon, adjacent_points
        )

    def calculate_distance_from_shoreline(self, mesh_vector: MeshVector) -> None:
        """
        Calculate the distance from the shoreline for a given mesh vector.

        Parameters
        ----------
        mesh_vector : MeshVector
            The mesh vector containing latitude and longitude.

        Returns
        -------
        None
        """
        adjacent_points = AdjacentPoints.find_global_adjacent_points(
            self.offshore_distance_surface.lati,
            self.offshore_distance_surface.loni,
            mesh_vector.lat,
            mesh_vector.lon,
        )

        mesh_vector.distance_from_shoreline = interpolate_global_surface(
            self.offshore_distance_surface,
            mesh_vector.lat,
            mesh_vector.lon,
            adjacent_points,
        )
