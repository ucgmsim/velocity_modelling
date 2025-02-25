import logging
import sys
from logging import Logger
from typing import Dict, List

import numpy as np

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
            self.depths[i] = interpolate_global_surface(
                global_surf_read, mesh_vector, adjacent_points
            )

        # Find indices where top_val < bot_val
        mask = self.depths[:-1] < self.depths[1:]

        # Apply the condition using NumPy indexing
        self.depths[1:][mask] = self.depths[:-1][mask]


def interpolate_global_surface(
    global_surface_read: GlobalSurfaceRead,
    mesh_vector: MeshVector,
    adjacent_points: AdjacentPoints,
):
    """
    Interpolate the global surface value at a given latitude and longitude.

    Parameters
    ----------
    global_surface_read: GlobalSurfaceRead
        Object containing the global surface data.
    mesh_vector: MeshVector
        Object containing latitude and longitude.
    adjacent_points: AdjacentPoints
        Object containing indices of points adjacent to the lat-lon for interpolation.

    Returns
    -------
    float: Interpolated value at the given lat-lon.
    """
    # if point lies within the surface bounds, perform bilinear interpolation
    if adjacent_points.in_surface_bounds:

        x1 = global_surface_read.loni[adjacent_points.lon_ind[0]]
        x2 = global_surface_read.loni[adjacent_points.lon_ind[1]]

        y1 = global_surface_read.lati[adjacent_points.lat_ind[0]]
        y2 = global_surface_read.lati[adjacent_points.lat_ind[1]]

        q11 = global_surface_read.raster[adjacent_points.lon_ind[0]][
            adjacent_points.lat_ind[0]
        ]
        q12 = global_surface_read.raster[adjacent_points.lon_ind[0]][
            adjacent_points.lat_ind[1]
        ]
        q21 = global_surface_read.raster[adjacent_points.lon_ind[1]][
            adjacent_points.lat_ind[0]
        ]
        q22 = global_surface_read.raster[adjacent_points.lon_ind[1]][
            adjacent_points.lat_ind[1]
        ]

        assert x1 != x2
        assert y1 != y2

        q = bi_linear_interpolation(
            x1, x2, y1, y2, q11, q12, q21, q22, mesh_vector.lon, mesh_vector.lat
        )
        return q

    # if point lies within the extension zone, take on the value of the closest point
    elif adjacent_points.in_lat_extension_zone:
        p1 = global_surface_read.loni[adjacent_points.lon_ind[0]]
        p2 = global_surface_read.loni[adjacent_points.lon_ind[1]]
        v1 = global_surface_read.raster[adjacent_points.lon_ind[0]][
            adjacent_points.lat_edge_ind
        ]
        v2 = global_surface_read.raster[adjacent_points.lon_ind[1]][
            adjacent_points.lat_edge_ind
        ]
        p3 = mesh_vector.lon
        return linear_interpolation(p1, p2, v1, v2, p3)
    elif adjacent_points.in_lon_extension_zone:
        p1 = global_surface_read.lati[adjacent_points.lat_ind[0]]
        p2 = global_surface_read.lati[adjacent_points.lat_ind[1]]
        v1 = global_surface_read.raster[adjacent_points.lon_edge_ind][
            adjacent_points.lat_ind[0]
        ]
        v2 = global_surface_read.raster[adjacent_points.lon_edge_ind][
            adjacent_points.lat_ind[1]
        ]
        p3 = mesh_vector.lat
        return linear_interpolation(p1, p2, v1, v2, p3)
    elif adjacent_points.in_corner_zone:
        return global_surface_read.raster[adjacent_points.corner_lon_ind][
            adjacent_points.corner_lat_ind
        ]

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
        from velocity_modelling.cvm.global_model import interpolate_global_surface

        adjacent_points = AdjacentPoints.find_global_adjacent_points(
            self.vs30.lati, self.vs30.loni, mesh_vector.lat, mesh_vector.lon
        )

        mesh_vector.vs30 = interpolate_global_surface(
            self.vs30, mesh_vector, adjacent_points
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
        from velocity_modelling.cvm.global_model import interpolate_global_surface

        # Find the adjacent points for interpolation
        adjacent_points = AdjacentPoints.find_global_adjacent_points(
            self.offshore_distance_surface.lati,
            self.offshore_distance_surface.loni,
            mesh_vector.lat,
            mesh_vector.lon,
        )

        # Interpolate the global surfaces value at the given latitude and longitude
        mesh_vector.distance_from_shoreline = interpolate_global_surface(
            self.offshore_distance_surface, mesh_vector, adjacent_points
        )
