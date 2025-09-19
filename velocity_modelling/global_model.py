"""
Global Model Module

This module provides functionality for handling global surfaces and tomography data,
including interpolation and depth calculations.

"""

from pathlib import Path

import numpy as np
from numba import njit

from velocity_modelling.geometry import (
    AdjacentPoints,
    MeshVector,
)
from velocity_modelling.interpolate import (
    bi_linear_interpolation,
    linear_interpolation,
)
from velocity_modelling.velocity1d import VelocityModel1D


class GlobalSurfaceRead:
    """
    Class to store global surface data.

    Parameters
    ----------
    file_path : Path
        The file path of the global surface data.
    latitudes : np.ndarray
        The latitude values.
    longitudes : np.ndarray
        The longitude values.
    raster : np.ndarray
        The raster values.

    Attributes
    ----------
    file_path : Path
        The file path of the global surface data.
    lats : np.ndarray
        The latitude values.
    lons : np.ndarray
        The longitude values.
    raster : np.ndarray
        The raster values.
    max_lat : float
        The maximum latitude value.
    min_lat : float
        The minimum latitude value.
    max_lon : float
        The maximum longitude value.
    min_lon : float
        The minimum longitude value.
    """

    def __init__(
        self,
        file_path: Path,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        raster: np.ndarray,
    ):
        """
        Initialize the GlobalSurfaceRead.
        """
        self.file_path = file_path
        self.lats = latitudes
        self.lons = longitudes
        self.raster = raster
        self.max_lat = max(self.lats)
        self.min_lat = min(self.lats)
        self.max_lon = max(self.lons)
        self.min_lon = min(self.lons)

    @property
    def nlat(self) -> int:
        """
        Get the number of latitude points.

        Returns
        -------
        int
            Number of latitude points.
        """
        return len(self.lats)

    @property
    def nlon(self) -> int:
        """
        Get the number of longitude points.

        Returns
        -------
        int
            Number of longitude points.
        """
        return len(self.lons)


class PartialGlobalSurfaceDepths:
    """
    Class to store partial global surface depths.

    Parameters
    ----------
    n_surfaces : int
        The number of global surfaces.

    Attributes
    ----------
    depths : np.ndarray
        Array of global surface depths (in metres).

    """

    def __init__(self, n_surfaces: int):
        """
        Initialize the PartialGlobalSurfaceDepths.

        """
        self.depths = np.zeros(n_surfaces, dtype=np.float64)

    def find_global_submodel_ind_vectorized(self, depths: np.ndarray) -> np.ndarray:
        """
        Find the indices of the global sub-velocity model for an array of depths.

        Parameters
        ----------
        depths : np.ndarray
            Array of depths (in metres) to find the sub-velocity model indices for.

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

        # for each x in "depths", find the index "j" of self.depths (ie. global surface depths) that satisfies
        # self.depths[j] >= x > self.depths[j+1]
        # Here, "j" is the index of the global sub-velocity model
        # np.searchsorted finds the index where the element should be inserted to maintain order
        indices = np.searchsorted(self.depths[::-1], depths, side="left")
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
        global_surfaces: list[GlobalSurfaceRead],
        mesh_vector: MeshVector,
    ) -> None:
        """
        Interpolate the surface depths at the lat lon location given in mesh_vector.

        Parameters
        ----------
        global_surfaces : list[GlobalSurfaceRead]
            List containing global surface data
        mesh_vector : MeshVector
            Object containing a single lat lon point with one or more depths.

        """
        for i in range(len(global_surfaces)):
            adjacent_points = AdjacentPoints.find_global_adjacent_points(
                global_surfaces[i].lats,
                global_surfaces[i].lons,
                mesh_vector.lat,
                mesh_vector.lon,
            )

            self.depths[i] = interpolate_global_surface(
                global_surfaces[i], mesh_vector.lat, mesh_vector.lon, adjacent_points
            )

        mask = self.depths[:-1] < self.depths[1:]
        self.depths[1:][mask] = self.depths[:-1][mask]


@njit
def interpolate_global_surface_numba(
    lats: np.ndarray,
    lons: np.ndarray,
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
    lats : np.ndarray
        Array of latitudes for the raster.
    lons : np.ndarray
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
        x1 = lons[lon_ind[0]]
        x2 = lons[lon_ind[1]]
        y1 = lats[lat_ind[0]]
        y2 = lats[lat_ind[1]]

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
        p1 = lons[lon_ind[0]]
        p2 = lons[lon_ind[1]]
        v1 = raster[lon_ind[0], lat_edge_ind]
        v2 = raster[lon_ind[1], lat_edge_ind]
        # explicitly cast to float to prevent numba from regarding them as ndarray
        return linear_interpolation(float(p1), float(p2), float(v1), float(v2), lon)

    elif in_lon_extension_zone:
        p1 = lats[lat_ind[0]]
        p2 = lats[lat_ind[1]]
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
        surface.lats,
        surface.lons,
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
    """
    Class to store tomography data.

    Parameters
    ----------
    name : str
        The name of the tomography data.
    surf_depth : list[float]
        The list of surfaces depths.
    special_offshore_tapering : bool
        Flag for special offshore tapering.
    gtl : bool
        Flag indicating if the GTL model is applied.
    vs30 : GlobalSurfaceRead
        The vs30 surfaces data.
    surfaces : list[dict[str, GlobalSurfaceRead]]
        List of surfaces data for each velocity type.
    offshore_distance_surface : GlobalSurfaceRead
        The offshore distance surfaces data.
    offshore_basin_model_1d : VelocityModel1D
        The offshore 1D model data.

    Attributes
    ----------
    name : str
        The name of the tomography data.
    surf_depth : list[float]
        The list of surfaces depths.
    surfaces : list[dict[str, GlobalSurfaceRead]]
        List of surfaces data for each velocity type.
    tomography_loaded : bool
        Flag indicating if the tomography data has been loaded.
    special_offshore_tapering : bool
        Flag for special offshore tapering.
    gtl : bool
        Flag indicating if the GTL model is applied.
    smooth_boundary : None
        Placeholder for smooth boundary.
    vs30 : GlobalSurfaceRead
        The vs30 surfaces data.
    offshore_distance_surface : GlobalSurfaceRead
        The offshore distance surfaces data.
    offshore_basin_model_1d : VelocityModel1D
        The offshore 1D model data.
    tomography_loaded : bool
        Flag indicating if the tomography data has been loaded.

    """

    def __init__(
        self,
        name: str,
        surf_depth: list[float],
        special_offshore_tapering: bool,
        gtl: bool,
        vs30: GlobalSurfaceRead,
        surfaces: list[dict[str, GlobalSurfaceRead]],
        offshore_distance_surface: GlobalSurfaceRead,
        offshore_basin_model_1d: VelocityModel1D,
    ):
        """
        Initialize the TomographyData.

        """
        self.name = name
        self.surf_depth = surf_depth
        self.surfaces = surfaces
        self.special_offshore_tapering = special_offshore_tapering
        self.gtl = gtl
        self.smooth_boundary = None
        self.vs30 = vs30
        self.offshore_distance_surface = offshore_distance_surface
        self.offshore_basin_model_1d = offshore_basin_model_1d

    def calculate_vs30_from_tomo_vs30_surface(self, mesh_vector: MeshVector) -> None:
        """
        Calculate the Vs30 value at a given mesh vector.

        Parameters
        ----------
        mesh_vector : MeshVector
            The mesh vector containing latitude and longitude.

        """
        adjacent_points = AdjacentPoints.find_global_adjacent_points(
            self.vs30.lats, self.vs30.lons, mesh_vector.lat, mesh_vector.lon
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

        """
        adjacent_points = AdjacentPoints.find_global_adjacent_points(
            self.offshore_distance_surface.lats,
            self.offshore_distance_surface.lons,
            mesh_vector.lat,
            mesh_vector.lon,
        )

        mesh_vector.distance_from_shoreline = interpolate_global_surface(
            self.offshore_distance_surface,
            mesh_vector.lat,
            mesh_vector.lon,
            adjacent_points,
        )
