import yaml

import bisect
from enum import Enum
import importlib
import inspect
import numpy as np
from typing import List, Dict
from pathlib import Path

from logging import Logger
import logging
import sys

from qcore import point_in_polygon
from qcore import coordinates

from velocity_modelling.cvm.constants import (
    MAX_LAT_SURFACE_EXTENSION,
    MAX_LON_SURFACE_EXTENSION,
)
from velocity_modelling.cvm.coordinates import lat_lon_to_distance
from velocity_modelling.cvm.interpolate import (
    bi_linear_interpolation,
    linear_interpolation,
)

DATA_ROOT = Path(__file__).parent.parent / "Data"
nzvm_registry_path = DATA_ROOT / "nzvm_registry.yaml"

DEFAULT_OFFSHORE_1D_MODEL = "Cant1D_v2"  # vm1d name for offshore 1D model
DEFAULT_OFFSHORE_DISTANCE = "offshore"  # surface name for offshore distance


class CVMRegistry:  # Forward declaration
    pass


class PartialGlobalSurfaceDepths:
    pass


class GlobalSurfaces:
    pass


class MeshVector:
    pass


class GlobalSurfaceRead:
    pass


import numpy as np


def interpolate_global_surface_depths(
    partial_global_surface_depth: PartialGlobalSurfaceDepths,
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
        Object containing calculation data and output directory.
    """
    for i in range(len(global_surfaces.surface)):
        global_surf_read = global_surfaces.surface[i]
        adjacent_points = global_surf_read.find_global_adjacent_points(mesh_vector)
        partial_global_surface_depth.depth[i] = interpolate_global_surface(
            global_surf_read, mesh_vector, adjacent_points
        )

    depths = partial_global_surface_depth.depth

    # Find indices where top_val < bot_val
    mask = depths[:-1] < depths[1:]

    # Apply the condition using NumPy indexing
    depths[1:][mask] = depths[:-1][mask]


class AdjacentPoints:
    pass


def interpolate_global_surface(
    global_surface_read: GlobalSurfaceRead,
    mesh_vector: MeshVector,
    adjacent_points: AdjacentPoints,
):
    """
    Interpolate the global surface value at a given latitude and longitude.

    Parameters:
    lat (float): Latitude of the point for interpolation.
    lon (float): Longitude of the point for interpolation.
    adjacent_points (AdjacentPoints): Object containing indices of points adjacent to the lat-lon for interpolation.

    Returns:
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

    def find_basin_adjacent_points(self, mesh_vector: MeshVector):
        """
        Find the adjacent point to the mesh vector in the basin surface.

        Parameters
        ----------
        mesh_vector : MeshVector
            The mesh vector.

        Returns
        -------
        AdjacentPoints
            The indices of the adjacent points.
        """
        lat = mesh_vector.lat
        lon = mesh_vector.lon

        lat_assigned_flag = False
        lon_assigned_flag = False

        adjacent_points = AdjacentPoints()

        # Handle latitude
        if self.lati[0] > self.lati[-1]:  # descending order
            lat_idx = np.searchsorted(self.lati[::-1], lat)
            lat_idx = len(self.lati) - lat_idx - 1
        else:  # ascending order
            lat_idx = np.searchsorted(self.lati, lat)

        if 0 < lat_idx < len(self.lati):
            adjacent_points.lat_ind = [lat_idx - 1, lat_idx]
        elif lat_idx == 0 and self.lati[0] == lat:
            adjacent_points.lat_ind = [0, 1]
        elif lat_idx == len(self.lati) and self.lati[-1] == lat:
            adjacent_points.lat_ind = [len(self.lati) - 1, len(self.lati)]

        # Handle longitude
        if self.loni[0] > self.loni[-1]:  # descending order
            lon_idx = np.searchsorted(self.loni[::-1], lon)
            lon_idx = len(self.loni) - lon_idx - 1
        else:  # ascending order
            lon_idx = np.searchsorted(self.loni, lon)

        if 0 < lon_idx < len(self.loni):
            adjacent_points.lon_ind = [lon_idx - 1, lon_idx]
        elif lon_idx == 0 and self.loni[0] == lon:
            adjacent_points.lon_ind = [0, 1]
        elif lon_idx == len(self.loni) and self.loni[-1] == lon:
            adjacent_points.lon_ind = [len(self.loni) - 1, len(self.loni)]

        if adjacent_points.lat_ind != [0, 0] and adjacent_points.lon_ind != [0, 0]:
            adjacent_points.in_surface_bounds = True
        else:
            print(
                f"Error, basin point lies outside of the extent of the basin surface ({lon}, {lat})."
            )

        return adjacent_points


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


class PartialGlobalSurfaceDepths:
    def __init__(self, n_surface: int):
        """
        Initialize the PartialGlobalSurfaceDepth.

        Parameters
        ----------
        n_surface : int
            The number of global surfaces.
        """
        self.depth = np.zeros(n_surface, dtype=np.float64)


class PartialGlobalMesh:
    def __init__(self, nx: int, nz: int):
        """
        Initialize the PartialGlobalMesh.

        Parameters
        ----------
        nx : int
            The number of points in the X direction.
        nz : int
            The number of points in the Z direction.
        """
        self.lon = np.zeros(nx)
        self.lat = np.zeros(nx)
        self.x = np.zeros(nx)
        self.z = np.zeros(nz)
        # self.nx = nx
        # self.ny = 1
        # self.nz = nz
        self.y = 0.0

    @property
    def nx(self):
        return len(self.x)

    @property
    def ny(self):
        return len(self.y)

    @property
    def nz(self):
        return len(self.z)


class MeshVector:
    def __init__(self, nz, lat=None, lon=None):
        self.lat = lat
        self.lon = lon
        self.z = np.zeros(nz)
        # self.nz = nz
        self.vs30 = None
        self.distance_from_shoreline = None

    @property
    def nz(self):
        return len(self.z)


class PartialGlobalQualities:
    def __init__(self, lon_grid_dim_max: int, dep_grid_dim_max: int):
        self.vp = np.zeros(
            (lon_grid_dim_max, dep_grid_dim_max), dtype=np.float64
        )  # TODO: why dim max??
        self.vs = np.zeros((lon_grid_dim_max, dep_grid_dim_max), dtype=np.float64)
        self.rho = np.zeros((lon_grid_dim_max, dep_grid_dim_max), dtype=np.float64)
        self.inbasin = np.zeros((lon_grid_dim_max, dep_grid_dim_max), dtype=np.int8)


class GlobalSurfaces:
    def __init__(self):
        """
        Initialize the GlobalSurfaces.
        """
        self.surface = []


class GlobalSurfaceRead:
    def __init__(self, nlat: int, nlon: int):
        """
        Initialize the GlobalSurfaceRead.

        Parameters
        ----------
        nlat : int
            The number of latitude points.
        nlon : int
            The number of longitude points.
        """
        #    self.nlat = nlat
        #    self.nlon = nlon
        self.lati = np.zeros(nlat)
        self.loni = np.zeros(nlon)
        self.raster = np.zeros((nlon, nlat))
        self.max_lat = None
        self.min_lat = None
        self.max_lon = None
        self.min_lon = None

    @property
    def nlat(self):
        return len(self.lati)

    @property
    def nlon(self):
        return len(self.loni)

    def find_global_adjacent_points(self, mesh_vector: MeshVector):
        """
        Find the adjacent points to the mesh vector in the global surface.

        Parameters
        ----------
        mesh_vector : MeshVector
            The mesh vector.

        Returns
        -------
        AdjacentPoints
            The indices of the adjacent points.
        """

        lat = mesh_vector.lat
        lon = mesh_vector.lon

        lat_assigned_flag = False
        lon_assigned_flag = False

        adjacent_points = AdjacentPoints()

        if self.lati[0] < lat <= self.lati[-1] or self.lati[0] > lat >= self.lati[-1]:
            is_ascending = self.lati[0] < self.lati[-1]
            lati = (
                self.lati if is_ascending else self.lati[::-1]
            )  # reverse the array if sorted in descending order

            index = np.searchsorted(lati, lat)
            index = (
                self.nlat - index if not is_ascending else index
            )  # reverse the index if sorted in descending order

            if 0 < index < self.nlat:
                adjacent_points.lat_ind[0] = index - 1
                adjacent_points.lat_ind[1] = index
                lat_assigned_flag = True

        if self.loni[0] < lon <= self.loni[-1] or self.loni[0] > lon >= self.loni[-1]:
            is_ascending = self.loni[0] < self.loni[-1]
            loni = (
                self.loni if is_ascending else self.loni[::-1]
            )  # reverse the array if sorted in descending order

            index = np.searchsorted(loni, lon)
            index = (
                self.nlon - index if not is_ascending else index
            )  # reverse the index if sorted in descending order

            if 0 < index < self.nlon:
                adjacent_points.lon_ind[0] = index - 1
                adjacent_points.lon_ind[1] = index
                lon_assigned_flag = True

        if lat_assigned_flag and lon_assigned_flag:
            adjacent_points.in_surface_bounds = True
        else:
            if lon_assigned_flag and not lat_assigned_flag:
                if (
                    lat - self.max_lat
                ) <= MAX_LAT_SURFACE_EXTENSION and lat >= self.max_lat:
                    adjacent_points.in_lat_extension_zone = True
                    self.find_edge_inds(adjacent_points, 1)
                elif (
                    self.min_lat - lat
                ) <= MAX_LAT_SURFACE_EXTENSION and lat <= self.min_lat:
                    adjacent_points.in_lat_extension_zone = True
                    self.find_edge_inds(adjacent_points, 3)
            if lat_assigned_flag and not lon_assigned_flag:
                if (
                    self.min_lon - lon
                ) <= MAX_LON_SURFACE_EXTENSION and lon <= self.min_lon:
                    adjacent_points.in_lon_extension_zone = True
                    self.find_edge_inds(adjacent_points, 4)
                elif (
                    lon - self.max_lon
                ) <= MAX_LON_SURFACE_EXTENSION and lon >= self.max_lon:
                    adjacent_points.in_lon_extension_zone = True
                    self.find_edge_inds(adjacent_points, 2)
            # four cases for corner zones
            if (
                (lat - self.max_lat) <= MAX_LAT_SURFACE_EXTENSION
                and (self.min_lon - lon) <= MAX_LON_SURFACE_EXTENSION
                and lon <= self.min_lon
                and lat >= self.max_lat
            ):
                self.find_corner_inds(self.max_lat, self.min_lon, adjacent_points)
            elif (
                lat - self.max_lat <= MAX_LAT_SURFACE_EXTENSION
                and lon - self.max_lon <= MAX_LON_SURFACE_EXTENSION
                and lon >= self.max_lon
                and lat >= self.max_lat
            ):
                self.find_corner_inds(self.max_lat, self.max_lon, adjacent_points)
            elif (
                self.min_lat - lat <= MAX_LAT_SURFACE_EXTENSION
                and self.min_lon - lon <= MAX_LON_SURFACE_EXTENSION
                and lon <= self.min_lon
                and lat <= self.min_lat
            ):
                self.find_corner_inds(self.min_lat, self.min_lon, adjacent_points)
            elif (
                self.min_lat - lat <= MAX_LAT_SURFACE_EXTENSION
                and lon - self.max_lon <= MAX_LON_SURFACE_EXTENSION
                and lon >= self.max_lon
                and lat <= self.min_lat
            ):
                self.find_corner_inds(self.min_lat, self.max_lon, adjacent_points)

            if not (
                adjacent_points.in_lat_extension_zone
                or adjacent_points.in_lon_extension_zone
                or adjacent_points.in_corner_zone
            ):
                raise ValueError(
                    f"Point does not lie in any global surface extension. {lon} {lat}"
                )

        return adjacent_points

    def find_edge_inds(self, adjacent_points, edge_type):
        """
        Find the indices of the edge of the global surface closest to the lat-lon point.

        Parameters:
        adjacent_points (AdjacentPoints): Object containing indices of points adjacent to the lat-lon for interpolation.
        edge_type (int): Indicating whether the point lies to the north, east, south, or west of the global surface.
        """
        nlat = self.nlat
        nlon = self.nlon

        if edge_type == 1:  # what does each edge_type mean?
            if self.max_lat == self.lati[0]:
                adjacent_points.lat_edge_ind = 0
            elif self.max_lat == self.lati[-1]:
                adjacent_points.lat_edge_ind = nlat - 1
            else:
                raise ValueError("Point lies outside of surface bounds.")
        elif edge_type == 3:
            if self.min_lat == self.lati[0]:
                adjacent_points.lat_edge_ind = 0
            elif self.min_lat == self.lati[-1]:
                adjacent_points.lat_edge_ind = nlat - 1
            else:
                raise ValueError("Point lies outside of surface bounds.")
        elif edge_type == 2:
            if self.max_lon == self.loni[0]:
                adjacent_points.lon_edge_ind = 0
            elif self.max_lon == self.loni[-1]:
                adjacent_points.lon_edge_ind = nlon - 1
            else:
                raise ValueError("Point lies outside of surface bounds.")
        elif edge_type == 4:
            if self.min_lon == self.loni[0]:
                adjacent_points.lon_edge_ind = 0
            elif self.min_lon == self.loni[-1]:
                adjacent_points.lon_edge_ind = nlon - 1
            else:
                raise ValueError("Point lies outside of surface bounds.")
        else:
            raise ValueError("Point lies outside of surface bounds.")

    def find_corner_inds(self, lat_pt, lon_pt, adjacent_points: AdjacentPoints):
        """
        Find the indices of the corner of the global surface closest to the lat-lon point.

        Parameters:
        lat_pt (float): Latitude of point for eventual interpolation.
        lon_pt (float): Longitude of point for eventual interpolation.
        adjacent_points (AdjacentPoints): Object containing indices of points adjacent to the lat-lon for interpolation.
        """
        nlat = self.nlat
        nlon = self.nlon

        if np.isclose(lat_pt, self.lati[0]):
            adjacent_points.corner_lat_ind = 0
        elif np.isclose(lat_pt, self.lati[-1]):
            adjacent_points.corner_lat_ind = nlat - 1
        else:
            raise ValueError(
                f"Point lies outside of surface bounds. Lat {lat_pt} outside [{self.lati[0]} {self.lati[-1]}]"
            )

        if np.isclose(lon_pt, self.loni[0]):
            adjacent_points.corner_lon_ind = 0
        elif np.isclose(lon_pt, self.loni[-1]):
            adjacent_points.corner_lon_ind = nlon - 1
        else:
            raise ValueError(
                f"Point lies outside of surface bounds. Lon {lon_pt} outside [{self.loni[0]} {self.loni[-1]}]"
            )

        adjacent_points.in_corner_zone = True


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


class SmoothingBoundary:
    def __init__(self):
        """
        Initialize the SmoothingBoundary.
        """
        self.n = 0
        self.xpts = []
        self.ypts = []

    def determine_if_lat_lon_within_smoothing_region(self, mesh_vector: MeshVector):
        """
        Determine the closest index within the smoothing boundary to the given mesh vector coordinates.

        Parameters:
        smoothing_boundary (SmoothingBoundary): Object containing smoothing boundary data.
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
        smoothing_boundary (SmoothingBoundary): Object containing smoothing boundary data.
        mesh_vector (MeshVector): Object containing mesh vector data.

        Returns:
        int: Index of the closest point in the smoothing boundary.
        """
        boundary_points = np.column_stack((self.ypts, self.xpts))
        # the below wasn't giving the exact same results as the original code
        distances1 = coordinates.distance_between_wgs_depth_coordinates(
            boundary_points, np.array([mesh_vector.lat, mesh_vector.lon])
        )
        distances = (
            lat_lon_to_distance(boundary_points, mesh_vector.lat, mesh_vector.lon)
            * 1000
        )  # convert to meters
        closest_ind = np.argmin(distances)
        return closest_ind, distances[closest_ind]


class VeloMod1DData:
    def __init__(
        self, vp: np.ndarray, vs: np.ndarray, rho: np.ndarray, depth: np.ndarray
    ):
        """
        Initialize the VeloMod1DData.
        """
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.depth = depth
        self.n_depth = len(vp)  # maybe should be len(dep) but I'm not sure
        assert len(vp) == len(vs) == len(rho) == len(depth)


class VTYPE(Enum):
    vp = 0
    vs = 1
    rho = 2


class TomographyData:
    def __init__(
        self,
        cvm_registry: CVMRegistry,
        tomo_name: str,
        offshore_surface_name: str,
        offshore_v1d_name: str,
        logger: Logger = None,
    ):
        """
        Initialize the TomographyData.

        Parameters
        ----------
        cvm_registry : CVMRegistry
            The CVMRegistry instance.
        tomo_name : str
            The name of the tomography data.
        offshore_surface_name : str
            The name of the offshore surface.
        offshore_v1d_name : str
            The name of the offshore 1D model.
        logger : Logger, optional

        """
        tomo = cvm_registry.get_info("tomography", tomo_name)
        self.name = tomo_name

        self.surf_depth = tomo["elev"]
        self.surface = []

        self.tomography_loaded = False
        self.special_offshore_tapering = tomo["special_offshore_tapering"]
        self.smooth_boundary = SmoothingBoundary()

        surf_tomo_path = cvm_registry.get_full_path(tomo["path"])
        offshore_surface_path = cvm_registry.get_info("surface", offshore_surface_name)[
            "path"
        ]
        offshore_v1d_path = cvm_registry.get_info("vm1d", offshore_v1d_name)["path"]

        self.vs30 = cvm_registry.load_global_surface(
            tomo["vs30_path"]
        )  # GlobalSurfaceRead

        for i, elev in enumerate(self.surf_depth):
            self.surface.append({})
            elev_name = (
                f"{elev}" if elev == int(elev) else f"{elev:.2f}".replace(".", "p")
            )
            for vtype in VTYPE:
                tomofile = (
                    surf_tomo_path / f"surf_tomography_{vtype.name}_elev{elev_name}.in"
                )
                assert tomofile.exists()
                self.surface[i][vtype.name] = cvm_registry.load_global_surface(tomofile)

        self.offshore_distance_surface = cvm_registry.load_global_surface(
            offshore_surface_path
        )
        self.offshore_basin_model_1d = cvm_registry.load_1d_velo_sub_model(
            offshore_v1d_path
        )
        self.tomography_loaded = True
        self.logger = logger

    def log(self, message, level=logging.INFO):
        if self.logger is not None:
            self.logger.log(level, message)
        else:
            print(message, file=sys.stderr)

    def calculate_vs30_from_tomo_vs30_surface(self, mesh_vector: MeshVector):

        adjacent_points = self.vs30.find_global_adjacent_points(mesh_vector)

        mesh_vector.vs30 = interpolate_global_surface(
            self.vs30, mesh_vector, adjacent_points
        )

    def calculate_distance_from_shoreline(self, mesh_vector: MeshVector):
        """
        Calculate the distance from the shoreline for a given mesh vector.

        Parameters:
        mesh_vector (MeshVector): The mesh vector containing latitude and longitude.

        Returns:
        None: The result is stored in the mesh_vector's distance_from_shoreline attribute.
        """

        # Find the adjacent points for interpolation
        adjacent_points = self.offshore_distance_surface.find_global_adjacent_points(
            mesh_vector,
        )

        # Interpolate the global surface value at the given latitude and longitude
        mesh_vector.distance_from_shoreline = interpolate_global_surface(
            self.offshore_distance_surface, mesh_vector, adjacent_points
        )


def check_boundary_index(func):
    def wrapper(self, i, *args, **kwargs):
        if i < 0 or i >= len(self.boundary):
            self.log(
                f"Error: basin boundary {i} not found. Max index is {len(self.boundary) - 1}"
            )
            return None
        return func(self, i, *args, **kwargs)

    return wrapper


class BasinData:  # forward declaration
    pass


class InBasin:
    def __init__(self, basin_data: BasinData, dep_grid_dim: int):
        self.basin_data = basin_data
        self.in_basin_lat_lon = np.zeros(len(basin_data.boundary), dtype=bool)
        self.in_basin_depth = np.zeros((dep_grid_dim), dtype=bool)


class PartialBasinSurfaceDepths:
    def __init__(self, basin_data: BasinData):
        # List of arrays of depths for each surface of the basin
        # self.depth[i] is the depth of the i-th surface
        self.depth = np.zeros(len(basin_data.surface), dtype=np.float64)


class QualitiesVector:
    pass


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

        self.boundary = [
            cvm_registry.load_basin_boundary(boundary_path)
            for boundary_path in basin_info["boundaries"]
        ]
        self.surface = [
            cvm_registry.load_basin_surface(surface)
            for surface in basin_info["surfaces"]
        ]
        self.submodel = [
            cvm_registry.load_basin_submodel(surface)
            for surface in basin_info["surfaces"]
        ]

        self.perturbation_data = None
        self.logger = logger

        self.log(f"Basin {basin_name} fully loaded.")

    def log(self, message, level=logging.INFO):
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
        return self.boundary[i][:, 1]

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
        return self.boundary[i][:, 0]

    @check_boundary_index
    def min_lon_boundary(self, i: int) -> float:
        """
        Get the minimum longitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        float
            The minimum longitude of the boundary.
        """
        return np.min(self.boundary_lon(i))

    @check_boundary_index
    def max_lon_boundary(self, i: int) -> float:
        """
        Get the maximum longitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        float
            The maximum longitude of the boundary.
        """
        return np.max(self.boundary_lon(i))

    @check_boundary_index
    def min_lat_boundary(self, i: int) -> float:
        """
        Get the minimum latitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        float
            The minimum latitude of the boundary.
        """
        return np.min(self.boundary_lat(i))

    @check_boundary_index
    def max_lat_boundary(self, i: int) -> float:
        """
        Get the maximum latitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        float
            The maximum latitude of the boundary.
        """
        return np.max(self.boundary_lat(i))

    def point_on_vertex(self, boundary_ind: int, mesh_vector: MeshVector) -> bool:
        """
        Check if a point lies on a vertex of a basin boundary.

        Parameters
        ----------
        boundary_ind : int
            The index of the boundary.
        mesh_vector : MeshVector
            The mesh vector containing the point.

        Returns
        -------
        bool
            True if the point lies on a vertex of the boundary, False otherwise.
        """
        boundary_lats = self.boundary_lat(boundary_ind)
        boundary_lons = self.boundary_lon(boundary_ind)
        on_vertex = np.any(
            np.isclose(boundary_lats, mesh_vector.lat)
            & np.isclose(boundary_lons, mesh_vector.lon)
        )
        return on_vertex

    def determine_if_within_basin_lat_lon(
        self, mesh_vector: MeshVector, in_basin: InBasin = None
    ):
        """
        Determine if a point lies within the different basin boundaries.

        Parameters:
        basin_data (BasinData): Struct containing basin data (surfaces, submodels, etc.)
        global_model_parameters (GlobalModelParameters): Struct containing all model parameters (surface names, submodel names, basin names, etc.)
        lat (float): Latitude value of point of concern
        lon (float): Longitude value of point of concern

        Returns:
        int: 1 if inside a basin (any), 0 otherwise
        """

        # TODO: Only Perturbation basins are ignored for smoothing, which will be handled in the perturbation code.
        #  We dropped ignoreBasinForSmoothing from Basin definition. By default we don't ignore any basins for smoothing.

        # See https://github.com/ucgmsim/mapping/blob/80b8e66222803d69e2f8f2182ccc1adc467b7cb1/mapbox/vs30/scripts/basin_z_values/gen_sites_in_basin.py#L119C2-L123C55
        # and https://github.com/ucgmsim/qcore/blob/master/qcore/point_in_polygon.py

        on_vertex = False

        for ind, boundary in enumerate(self.boundary):

            if not (
                np.min(boundary[:, 0]) <= mesh_vector.lon <= np.max(boundary[:, 0])
                and np.min(boundary[:, 1]) <= mesh_vector.lat <= np.max(boundary[:, 1])
            ):
                continue  # outside of basin

            else:
                # possibly in basin
                in_poly = point_in_polygon.is_inside_postgis(
                    boundary, np.array([mesh_vector.lon, mesh_vector.lat])
                )  # check if in poly

                if in_poly:  # in_poly == 1 (inside) ==2 (on edge)
                    if in_basin and type(in_basin) == InBasin:
                        in_basin.in_basin_lat_lon[ind] = True
                    return True  # inside a basin (any)
                else:  # outside poly
                    if (
                        in_basin and type(in_basin) == InBasin
                    ):  # check if it is on vertex. if in_poly
                        in_basin.in_basin_lat_lon[ind] = self.point_on_vertex(
                            ind, mesh_vector
                        )

                    continue  # outside of basin

        return False  # not inside basin

    def determine_basin_surface_depths(
        self,
        in_basin: InBasin,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        mesh_vector: MeshVector,
    ):
        """
        Determine the basin surface depths for a given latitude and longitude.

        Parameters
        ----------
        in_basin : InBasin
            Struct containing flags to indicate if lat-lon point - depths lie within the basin.
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Struct containing depths for all applicable basin surfaces at one lat-lon location.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        """
        for surface_ind, surface in enumerate(self.surface):

            if np.any(
                in_basin.in_basin_lat_lon
            ):  # see if this is in any boundary of this basin
                adjacent_points = surface.find_basin_adjacent_points(mesh_vector)
                # TODO: check if in_surface_bounds is True
                x1 = surface.loni[adjacent_points.lon_ind[0]]
                x2 = surface.loni[adjacent_points.lon_ind[1]]
                y1 = surface.lati[adjacent_points.lat_ind[0]]
                y2 = surface.lati[adjacent_points.lat_ind[1]]
                q11 = surface.raster[adjacent_points.lon_ind[0]][
                    adjacent_points.lat_ind[0]
                ]
                q12 = surface.raster[adjacent_points.lon_ind[0]][
                    adjacent_points.lat_ind[1]
                ]

                q21 = surface.raster[adjacent_points.lon_ind[1]][
                    adjacent_points.lat_ind[0]
                ]

                q22 = surface.raster[adjacent_points.lon_ind[1]][
                    adjacent_points.lat_ind[1]
                ]
                partial_basin_surface_depths.depth[surface_ind] = (
                    bi_linear_interpolation(
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
                )
            else:
                partial_basin_surface_depths.depth[surface_ind] = None

    def enforce_surface_depths(
        self,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
    ):
        """
        Enforce the depths of the surfaces are consistent with stratigraphy.

        Parameters
        ----------

        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Depths for all applicable basin surfaces at one lat - lon location.

        Returns
        -------
        None
        """
        top_val, bot_val = None, None
        nan_obtained = False
        nan_ind = 0

        # for boundary_ind, surface in enumerate(self.surface):
        #     for i in range(len(surface) - 1, 0, -1):
        #         top_val = partial_basin_surface_depths.depth[boundary_ind][i - 1]
        #         bot_val = partial_basin_surface_depths.depth[boundary_ind][i]
        #
        #         if np.isnan(top_val):
        #             nan_obtained = True
        #             nan_ind = i
        #             break
        #         elif top_val < bot_val:
        #             partial_basin_surface_depths.depth[boundary_ind][i - 1] = bot_val
        #
        #     if nan_obtained:
        #         for j in range(nan_ind - 1):
        #             top_val = partial_basin_surface_depths.depth[boundary_ind][j]
        #             bot_val = partial_basin_surface_depths.depth[boundary_ind][j + 1]
        #             if top_val < bot_val:
        #                 partial_basin_surface_depths.depth[boundary_ind][j] = bot_val

        depths = partial_basin_surface_depths.depth
        # Find the first NaN value
        nan_indices = np.where(np.isnan(depths))[0]
        if nan_indices.size > 0:
            nan_ind = nan_indices[0]
        else:
            nan_ind = len(depths)
        # Enforce stratigraphy for depths before the first NaN
        for i in range(nan_ind - 1, 0, -1):
            if depths[i - 1] < depths[i]:
                depths[i - 1] = depths[i]
        # Enforce stratigraphy for depths after the first NaN
        for i in range(nan_ind - 1):
            if depths[i] < depths[i + 1]:
                depths[i] = depths[i + 1]

    def assign_basin_qualities(
        self,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        qualities_vector: QualitiesVector,
        nz_tomography_data: TomographyData,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon: bool,
        on_boundary: bool,
        depth: float,
        basin_num: int,
        z_ind: int,
    ):
        """
        Assign Vp, Vs, and Rho to the individual grid point.

        Parameters
        ----------
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Struct containing depths for all applicable basin surfaces at one lat-lon location.
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Struct containing global surface depths.
        nz_tomography_data : TomographyData
            Struct containing tomography data.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        in_any_basin_lat_lon : bool
            Flag indicating if the point is in any basin.
        on_boundary : bool
            Flag indicating if the point is on a boundary.
        depth : float
            The depth of the grid point to determine the properties at.
        basin_num : int
            The basin number pertaining to the basin of interest.
        z_ind : int
            The depth index of the single grid point.

        Returns
        -------
        None
        """
        # Determine the indices of the basin surfaces above and below the given depth
        ind_above = self.determine_basin_surface_above(
            partial_basin_surface_depths, depth
        )
        ind_below = self.determine_basin_surface_below(
            partial_basin_surface_depths, depth
        )

        # Call the sub-velocity models to assign the qualities
        self.call_basin_sub_velocity_models(
            partial_basin_surface_depths,
            partial_global_surface_depths,
            qualities_vector,
            nz_tomography_data,
            mesh_vector,
            in_any_basin_lat_lon,
            on_boundary,
            depth,
            ind_above,
            basin_num,
            z_ind,
        )

    def determine_basin_surface_above(
        self, partial_basin_surface_depths: PartialBasinSurfaceDepths, depth
    ):
        """
        Determine the index of the basin surface directly above the given depth.

        Parameters
        ----------
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Struct containing depths for all applicable basin surfaces at one lat-lon location.
        depth : float
            The depth of the grid point to determine the properties at.

        Returns
        -------
        int
            Index of the surface directly above the grid point.
        """
        # partial_basin_surface_depths.depth is a list of arrays of depths for each surface associated with each boundary
        #        for i in range(len(partial_basin_surface_depths.depth)-1,-1,-1):

        depths = partial_basin_surface_depths.depth  # depth is decreasing order

        valid_indices = np.where((~np.isnan(depths)) & (depths >= depth))[0]
        return valid_indices[-1] if valid_indices.size > 0 else 0  # the last one

    def determine_basin_surface_below(self, partial_basin_surface_depths, depth):
        """
        Determine the index of the basin surface directly below the given depth.

        Parameters
        ----------
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Struct containing depths for all applicable basin surfaces at one lat-lon location.
        depth : float
            The depth of the grid point to determine the properties at.

        Returns
        -------
        int
            Index of the surface directly below the grid point.
        """
        depths = partial_basin_surface_depths.depth
        valid_indices = np.where((~np.isnan(depths)) & (depths <= depth))[0]
        return valid_indices[-1] if valid_indices.size > 0 else 0  # the last index

    def enforce_basin_surface_depths(
        self,
        in_basin: InBasin,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        mesh_vector: MeshVector,
    ):
        """
        Enforce the depths of the surfaces are consistent with stratigraphy.

        Parameters
        ----------
        in_basin : InBasin
            Struct containing flags to indicate if lat-lon point - depths lie within the basin.
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Struct containing depths for all applicable basin surfaces at one lat-lon location.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.


        Returns
        -------
        None
        """

        if np.any(in_basin.in_basin_lat_lon):

            self.enforce_surface_depths(partial_basin_surface_depths)
            # TODO: check if this is correct
            top_lim = partial_basin_surface_depths.depth[
                0
            ]  # the depth of the first surface
            bot_lim = partial_basin_surface_depths.depth[
                -1
            ]  # the depth of the last surface

            in_basin.in_basin_depth = (bot_lim <= mesh_vector.z) & (
                mesh_vector.z <= top_lim
            )  # check if the point is within the basin
        else:
            in_basin.in_basin_depth.fill(False)

    def interpolate_basin_surface_depths(
        self,
        in_basin: InBasin,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        mesh_vector: MeshVector,
    ):
        """
        Determine if a lat-lon point is in a basin, if so interpolate the basin surface depths, enforce their hierarchy, then determine which depth points lie within the basin limits.

        Parameters
        ----------
        in_basin : InBasin
            Struct containing flags to indicate if lat-lon point - depths lie within the basin.
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Struct containing depths for all applicable basin surfaces at one lat-lon location.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        """
        self.determine_if_within_basin_lat_lon(mesh_vector, in_basin)
        self.determine_basin_surface_depths(
            in_basin, partial_basin_surface_depths, mesh_vector
        )
        self.enforce_basin_surface_depths(
            in_basin, partial_basin_surface_depths, mesh_vector
        )

    def call_basin_sub_velocity_models(
        self,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        qualities_vector: QualitiesVector,
        nz_tomography_data,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon,
        on_boundary,
        depth,
        ind_above,
        basin_num,
        z_ind,
    ):
        """
        Call the appropriate sub-velocity model based on the basin submodel name.

        Parameters
        ----------
        partial_basin_surface_depths : PartialBasinSurfaceDepths
            Struct containing depths for all applicable basin surfaces at one lat-lon location.
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Struct containing global surface depths.
        nz_tomography_data : TomographyData
            Struct containing tomography data.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        in_any_basin_lat_lon : bool
            Flag indicating if the point is in any basin.
        on_boundary : bool
            Flag indicating if the point is on a boundary.
        depth : float
            The depth of the grid point to determine the properties at.
        ind_above : int
            Index of the surface directly above the grid point.
        basin_num : int
            The basin number pertaining to the basin of interest.
        z_ind : int
            The depth index of the single grid point.

        Returns
        -------
        None
        """

        qualities_vector.inbasin[z_ind] = basin_num
        submodel_name, submodel_data = self.submodel[ind_above]

        if submodel_name == "NaNsubMod":
            self.nan_sub_mod(z_ind)

        elif submodel_name in ["Cant1D_v1", "Cant1D_v2", "Cant1D_v2_Pliocene_Enforced"]:
            from velocity_modelling.cvm.submodel import Cant1D_v1 as Cant1D_v1

            Cant1D_v1.main(z_ind, depth, qualities_vector, submodel_data)
        elif submodel_name == "PaleogeneSubMod_v1":
            from velocity_modelling.cvm.submodel import (
                PaleogeneSubMod_v1 as PaleogeneSubMod_v1,
            )

            PaleogeneSubMod_v1.main(z_ind, qualities_vector)
        elif submodel_name == "PlioceneSubMod_v1":
            from velocity_modelling.cvm.submodel import (
                PlioceneSubMod_v1 as PlioceneSubMod_v1,
            )

            PlioceneSubMod_v1.main(z_ind, qualities_vector)
        elif submodel_name == "MioceneSubMod_v1":
            from velocity_modelling.cvm.submodel import (
                MioceneSubMod_v1 as MioceneSubMod_v1,
            )

            MioceneSubMod_v1.main(z_ind, qualities_vector)
        elif submodel_name == "BPVSubMod_v1":
            from velocity_modelling.cvm.submodel import BPVSubMod_v1 as BPVSubMod_v1

            BPVSubMod_v1.main(z_ind, qualities_vector)
        elif submodel_name == "BPVSubMod_v2":
            from velocity_modelling.cvm.submodel import BPVSubMod_v2 as BPVSubMod_v2

            BPVSubMod_v2.main(z_ind, qualities_vector)
        elif submodel_name == "BPVSubMod_v3":
            from velocity_modelling.cvm.submodel import BPVSubMod_v3 as BPVSubMod_v3

            BPVSubMod_v3.main(
                z_ind, depth, qualities_vector, partial_basin_surface_depths
            )
        elif submodel_name == "BPVSubMod_v4":
            from velocity_modelling.cvm.submodel import BPVSubMod_v4 as BPVSubMod_v4

            BPVSubMod_v4.main(
                z_ind,
                depth,
                qualities_vector,
                partial_basin_surface_depths,
                partial_global_surface_depths,
            )
        else:
            raise ValueError(f"Error: Submodel {submodel_name} not found in registry.")


class QualitiesVector:
    def __init__(self, dep_grid_dim_max: int):
        self.vp = np.zeros(dep_grid_dim_max, dtype=np.float64)
        self.vs = np.zeros(dep_grid_dim_max, dtype=np.float64)
        self.rho = np.zeros(dep_grid_dim_max, dtype=np.float64)
        self.inbasin = np.zeros(dep_grid_dim_max, dtype=np.int8)

    def prescribe_velocities(
        self,
        global_model_parameters: dict,
        velo_mod_1d_data: VeloMod1DData,
        nz_tomography_data: TomographyData,
        global_surfaces: GlobalSurfaces,
        basin_data_list: List[BasinData],
        mesh_vector: MeshVector,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        partial_basin_surface_depths_list: List[PartialBasinSurfaceDepths],
        in_basin_list: List[InBasin],
        topo_type: str,
        on_boundary: bool,
        calculation_log: Logger,
    ):

        interpolate_global_surface_depths(
            partial_global_surface_depths, global_surfaces, mesh_vector, calculation_log
        )

        dZ = 0
        depth_change = 0

        shifted_mesh_vector = None

        in_any_basin_lat_lon = any(
            [
                basin_data.determine_if_within_basin_lat_lon(mesh_vector)
                for basin_data in basin_data_list
            ]
        )
        # TODO: test this
        if topo_type == "SQUASHED":
            shifted_mesh_vector = MeshVector(
                mesh_vector.nz, lat=mesh_vector.lat, lon=mesh_vector.lon
            )

            depth_change = -mesh_vector.z  # is this correct????
            shifted_mesh_vector.z = (
                partial_global_surface_depths.depth[1] - depth_change
            )
        # TODO: test this
        elif topo_type == "SQUASHED_TAPERED":
            dZ = mesh_vector.z[0] - mesh_vector.z[1]
            TAPER_DIST = 1.0
            shifted_mesh_vector = MeshVector(
                mesh_vector.nz, lat=mesh_vector.lat, lon=mesh_vector.lon
            )

            depth_change = -mesh_vector.z
            TAPER_VAL = np.where(
                (depth_change == 0)
                | (partial_global_surface_depths.depth[1] == 0)
                | (partial_global_surface_depths.depth[1] < 0),
                1.0,
                1.0
                - (
                    depth_change / (partial_global_surface_depths.depth[1] * TAPER_DIST)
                ),
            )
            TAPER_VAL = np.clip(TAPER_VAL, 0.0, None)
            shifted_mesh_vector.z = (
                partial_global_surface_depths.depth[1] * TAPER_VAL - depth_change
            )

        elif topo_type in ["BULLDOZED", "TRUE"]:
            shifted_mesh_vector = mesh_vector

        else:
            raise ValueError("User specified TOPO_TYPE not recognised, see readme.")

        for basin_ind, basin_data in enumerate(basin_data_list):
            basin_data.interpolate_basin_surface_depths(
                in_basin_list[basin_ind],
                partial_basin_surface_depths_list[basin_ind],
                shifted_mesh_vector,
            )

        basin_flag = False
        z = 0

        # TODO: maybe a place to do some parallelization
        for k in range(mesh_vector.nz):
            if topo_type in ["BULLDOZED", "TRUE"]:
                z = mesh_vector.z[k]
            elif topo_type in ["SQUASHED", "SQUASHED_TAPERED"]:
                z = shifted_mesh_vector.z[k]

            for i, basin_data in enumerate(basin_data_list):
                in_basin = in_basin_list[i]
                if in_basin.in_basin_depth[k]:
                    basin_flag = True
                    self.inbasin[k] = i  # basin number

                    basin_data.assign_basin_qualities(
                        partial_basin_surface_depths_list[i],
                        partial_global_surface_depths,
                        self,
                        nz_tomography_data,
                        mesh_vector,
                        in_any_basin_lat_lon,
                        on_boundary,
                        z,
                        i,  # basin_num
                        k,  # z_ind
                    )

            if not basin_flag:
                self.inbasin[k] = (
                    0  # This is incorrect in the original code. It should be -1. 0 indicates this is inside the 0th basin
                )
                velo_mod_ind = self.find_global_sub_velo_model_ind(
                    z, partial_global_surface_depths
                )

                velo_mod_name = global_model_parameters["submodels"][velo_mod_ind]
                self.call_sub_velocity_model(
                    velo_mod_name,
                    z,
                    k,
                    global_model_parameters,
                    partial_global_surface_depths,
                    velo_mod_1d_data,
                    nz_tomography_data,
                    mesh_vector,
                    in_any_basin_lat_lon,
                    on_boundary,
                )

            if z > partial_global_surface_depths.depth[1]:
                self.rho[k] = np.nan
                self.vp[k] = np.nan
                self.vs[k] = np.nan

            basin_flag = False

        if topo_type == "BULLDOZED":
            # for k in range(mesh_vector.nz):
            #     if mesh_vector.z[k] > 0:
            #         self.nan_sub_mod(k)
            mask = mesh_vector.z > 0

            self.rho[mask] = np.nan
            self.vp[mask] = np.nan
            self.vs[mask] = np.nan

    def find_global_sub_velo_model_ind(
        self,
        depth: np.float64,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
    ):
        """
        Find the index of the global sub-velocity model at the given depth.

        Parameters
        ----------
        depth : float
            The depth (in m) to find the sub-velocity model index at.
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Struct containing global surface depths.

        Returns
        -------
        int
            The index of the global sub-velocity model.
        """
        try:
            n_velo_ind = np.where(partial_global_surface_depths.depth >= depth)[0][-1]
            if n_velo_ind == len(partial_global_surface_depths.depth):
                raise ValueError("Error: depth not found in global sub-velocity model.")
        except IndexError:
            raise ValueError("Error: depth not found in global sub-velocity model.")

        return n_velo_ind

    def call_sub_velocity_model(
        self,
        submodel_name: str,
        z: float,
        k: int,
        global_model_parameters: dict,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        velo_mod_1d_data: VeloMod1DData,
        nz_tomography_data: TomographyData,
        mesh_vector: MeshVector,
        in_any_basin_lat_lon: bool,
        on_boundary: bool,
    ):
        """
        Call the appropriate sub-velocity model based on the global sub-velocity model index.

        Parameters
        ----------
        submodel_name : str
            The name of the global submodel.
        z : float
            The depth of the grid point to determine the properties at.
        k : int
            The depth index of the single grid point.
        global_model_parameters : dict
            Struct containing all model parameters (surface names, submodel names, basin names, etc.)
        partial_global_surface_depths : PartialGlobalSurfaceDepths
            Struct containing global surface depths.
        velo_mod_1d_data : VeloMod1DData
            Struct containing the 1D velocity model data.
        nz_tomography_data : TomographyData
            Struct containing tomography data.
        mesh_vector : MeshVector
            Struct containing a single lat-lon point with one or more depths.
        in_any_basin_lat_lon : bool
            Flag indicating if the point is in any basin.
        on_boundary : bool
            Flag indicating if the point is on a boundary.

        Returns
        -------
        None
        """

        if submodel_name == "NaNsubMod":
            self.nan_sub_mod(k)

        elif submodel_name == "EPtomo2010subMod":
            from velocity_modelling.cvm.submodel import EPtomo2010 as eptomo2010

            eptomo2010.main(
                k,
                z,
                self,
                mesh_vector,
                nz_tomography_data,
                partial_global_surface_depths,
                global_model_parameters["GTL"],
                in_any_basin_lat_lon,
                on_boundary,
            )

        elif submodel_name == "Cant1D_v1":
            from velocity_modelling.cvm.submodel import Cant1D_v1 as Cant1D_v1

            Cant1D_v1.main(k, z, self, velo_mod_1d_data)
        else:
            raise ValueError(f"Error: Submodel {submodel_name} not found in registry.")

    def nan_sub_mod(self, k: int):
        """
        Assign NaN values to the velocities and density at the given depth index.

        Parameters
        ----------
        k : int
            The depth index to assign NaN values to.
        """
        self.rho[k] = np.nan
        self.vp[k] = np.nan
        self.vs[k] = np.nan


class CVMRegistry:
    def __init__(
        self,
        version: str,
        registry_path: Path = nzvm_registry_path,
        logger: Logger = None,
    ):
        """
        Initialize the CVMRegistry.

        Parameters
        ----------
        version : str
            The version of the velocity model.
        registry_path : Path, optional
            The path to the registry file (default is nzvm_registry_path).
        """
        with open(nzvm_registry_path, "r") as f:
            self.registry = yaml.safe_load(f)
        self.version = version
        self.vm_global_params = None

        for vminfo in self.registry["vm"]:
            if str(vminfo["version"]) == version:
                self.vm_global_params = vminfo
                break

        self.logger = logger

    def log(self, message, level=logging.INFO):
        if self.logger is not None:
            self.logger.log(level, message)
        else:
            print(message, file=sys.stderr)

    def get_info(self, datatype: str, name: str) -> Dict:
        """
        Get information from the registry.

        Parameters
        ----------
        datatype : str
            The type of data to retrieve (e.g., 'tomography', 'surface').
        name : str
            The name of the data entry.

        Returns
        -------
        Dict
            The information dictionary for the specified data entry.
        """
        try:
            self.registry[datatype]
        except KeyError:
            self.log(f"Error: {datatype} not found in registry")
            return None

        for info in self.registry[datatype]:
            assert (
                "name" in info
            ), f"Error: This entry in {datatype} has no name defined."
            if info["name"] == name:
                return info
        self.log(f"Error: {name} for datatype {datatype} not found in registry")
        return None

    def get_full_path(self, relative_path: Path) -> Path:
        """
        Get the full path for a given relative path.

        Parameters
        ----------
        relative_path : Path
            The relative path to convert.

        Returns
        -------
        Path
            The full path.
        """
        return (
            DATA_ROOT / relative_path
            if not Path(relative_path).is_absolute()
            else Path(relative_path)
        )

    def load_1d_velo_sub_model(self, v1d_path: Path) -> VeloMod1DData:
        """
        Load a 1D velocity submodel into memory.

        Parameters
        ----------
        v1d_path : Path
            The path to the 1D velocity model file.

        Returns
        -------
        VeloMod1DData
            The loaded 1D velocity model data.
        """
        v1d_path = self.get_full_path(v1d_path)

        try:
            with open(v1d_path, "r") as file:
                next(file)
                data = np.loadtxt(file)
                velo_mod_1d_data = VeloMod1DData(
                    data[:, 0], data[:, 1], data[:, 2], data[:, 5]
                )

        except FileNotFoundError:
            self.log(f"Error 1D velocity model file {v1d_path} not found.")
            exit(1)

        return velo_mod_1d_data

    def load_basin_data(self, basin_names: List[str]):
        """
        Load all basin data into the basin_data structure.

        Parameters
        ----------
        basin_names : List[str]
            List of basin names to load.

        Returns
        -------
        List[BasinData]
            List of loaded basin data.
        """
        all_basin_data = []
        for basin_name in basin_names:
            basin_data = BasinData(self, basin_name)
            all_basin_data.append(basin_data)
        return all_basin_data

    def load_basin_boundary(self, basin_boundary_path: Path):
        """
        Load a basin boundary from a file.

        Parameters
        ----------
        basin_boundary_path : Path
            The path to the basin boundary file.

        Returns
        -------
        np.ndarray
            The loaded basin boundary data.
        """
        try:
            basin_boundary_path = self.get_full_path(basin_boundary_path)
            data = np.loadtxt(basin_boundary_path)
            lon = data[:, 0]
            lat = data[:, 1]
            boundary_data = np.column_stack((lon, lat))

            assert lon[-1] == lon[0]
            assert lat[-1] == lat[0]
        except FileNotFoundError:
            self.log(f"Error basin boundary file {basin_boundary_path} not found.")
            exit(1)
        except Exception as e:
            self.log(f"Error reading basin boundary file {basin_boundary_path}: {e}")
            exit(1)
        return boundary_data

    def load_basin_submodel(self, basin_surface: dict):
        """
        Load a basin sub-model into the basin_data structure.

        Parameters
        ----------
        basin_surface : dict {'name': str, 'submodel': str}

            Dictionary containing basin surface data.

        Returns
        -------
        VeloMod1DData or None
            The loaded sub-model data or None if not applicable.
        """
        submodel_name = basin_surface["submodel"]
        if submodel_name == "null":
            return None
        submodel = self.get_info("submodel", submodel_name)

        if submodel is None:
            self.log(f"Error: submodel {submodel_name} not found.")
            exit(1)

        if submodel["type"] == "vm1d":
            vm1d = self.get_info("vm1d", submodel["name"])
            if vm1d is None:
                self.log(f"Error: vm1d {submodel['name']} not found.")
                exit(1)
            return (submodel_name, self.load_1d_velo_sub_model(vm1d["path"]))

        elif submodel["type"] == "relation":
            return (submodel_name, None)  # TODO: Implement relation submodel
        elif submodel["type"] == "perturbation":
            return (submodel_name, None)  # TODO: Implement perturbation submodel

    def load_basin_surface(self, basin_surface: str):
        """
        Load a basin surface from a file.

        Parameters
        ----------
        basin_surface : dict {'name': str, 'submodel': str}

        Returns
        -------
        BasinSurfaceRead
            The loaded basin surface data.
        """

        surface_info = self.get_info("surface", basin_surface["name"])

        self.log(f"Loading basin surface file {surface_info['path']}")

        basin_surface_path = self.get_full_path(surface_info["path"])

        try:
            with open(basin_surface_path, "r") as f:
                nlat, nlon = map(int, f.readline().split())
                basin_surf_read = BasinSurfaceRead(nlat, nlon)

                latitudes = np.fromfile(f, dtype=float, count=nlat, sep=" ")
                longitudes = np.fromfile(f, dtype=float, count=nlon, sep=" ")

                basin_surf_read.lati = latitudes
                basin_surf_read.loni = longitudes

                raster_data = np.fromfile(f, dtype=float, count=nlat * nlon, sep=" ")
                if len(raster_data) != nlat * nlon:
                    print(
                        f"Error: in {basin_surface_path} raster data length mismatch: {len(raster_data)} != {nlat * nlon}"
                    )
                    raster_data = np.pad(
                        raster_data, (0, nlat * nlon - len(raster_data)), "constant"
                    )

                basin_surf_read.raster = raster_data.reshape((nlat, nlon)).T

                first_lat = basin_surf_read.lati[0]
                last_lat = basin_surf_read.lati[nlat - 1]
                basin_surf_read.max_lat = max(first_lat, last_lat)
                basin_surf_read.min_lat = min(first_lat, last_lat)

                first_lon = basin_surf_read.loni[0]
                last_lon = basin_surf_read.loni[nlon - 1]
                basin_surf_read.max_lon = max(first_lon, last_lon)
                basin_surf_read.min_lon = min(first_lon, last_lon)

                return basin_surf_read

        except FileNotFoundError:
            self.log(f"Error basin surface file {basin_surface_path} not found.")
            exit(1)
        except Exception as e:
            self.log(f"Error: {e}")
            exit(1)

    def load_tomo_surface_data(
        self,
        tomo_name: str,
        offshore_surface_name: str = DEFAULT_OFFSHORE_DISTANCE,
        offshore_v1d_name: str = DEFAULT_OFFSHORE_1D_MODEL,
    ) -> TomographyData:
        """
        Load tomography surface data.

        Parameters
        ----------
        tomo_name : str
            The name of the tomography data.
        offshore_surface_name : str, optional
            The name of the offshore surface (default is DEFAULT_OFFSHORE_DISTANCE).
        offshore_v1d_name : str, optional
            The name of the offshore 1D model (default is DEFAULT_OFFSHORE_1D_MODEL).

        Returns
        -------
        TomographyData
            The loaded tomography data.
        """
        return TomographyData(self, tomo_name, offshore_surface_name, offshore_v1d_name)

    def load_all_global_data(self, logger: Logger):
        """
        Load all data required to generate the velocity model, global surfaces, sub velocity models, and all basin data.

        Parameters
        ----------
        logger : Logger
            Logger for logging information.

        Returns
        -------
        Tuple[VeloMod1DData, TomographyData, GlobalSurfaces, List[BasinData]]
            The loaded global data.
        """
        velo_mod_1d_data = None
        nz_tomography_data = None

        global_model_params = self.vm_global_params

        self.log("Loading global velocity submodel data.")
        for i in range(len(global_model_params["submodels"])):
            if global_model_params["submodels"][i] == "v1DsubMod":
                velo_mod_1d_data = self.load_1d_velo_sub_model(
                    global_model_params["submodels"][i]
                )
                self.log("Loaded 1D velocity model data.")
            elif global_model_params["submodels"][i] == "NaNsubMod":
                pass
            else:
                nz_tomography_data = self.load_tomo_surface_data(
                    global_model_params["tomography"]
                )
                self.log("Loaded tomography data.")

        if nz_tomography_data is not None:
            self.load_smooth_boundaries(
                nz_tomography_data, global_model_params["basins"]
            )

        self.log("Completed loading of global velocity submodel data.")

        global_surfaces = self.load_global_surface_data(global_model_params["surfaces"])
        self.log("Completed loading of global surfaces.")

        self.log("Loading basin data.")
        basin_data = self.load_basin_data(global_model_params["basins"])
        self.log("Completed loading basin data.")
        self.log("All global data loaded.")
        return velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data

    def load_global_surface(self, surface_file: Path):
        """
        Load a global surface from a file.

        Parameters
        ----------
        surface_file : Path
            The path to the global surface file.

        Returns
        -------
        GlobalSurfaceRead
            The loaded global surface data.
        """
        surface_file = self.get_full_path(surface_file)

        try:
            with open(surface_file, "r") as f:
                nlat, nlon = map(int, f.readline().split())
                global_surf_read = GlobalSurfaceRead(nlat, nlon)

                latitudes = np.fromfile(f, dtype=float, count=nlat, sep=" ")
                longitudes = np.fromfile(f, dtype=float, count=nlon, sep=" ")

                global_surf_read.lati = latitudes
                global_surf_read.loni = longitudes

                raster_data = np.fromfile(f, dtype=float, count=nlat * nlon, sep=" ")
                if len(raster_data) != nlat * nlon:
                    self.log(
                        f"Error: in {surface_file} raster data length mismatch: {len(raster_data)} != {nlat * nlon}"
                    )
                    raster_data = np.pad(
                        raster_data, (0, nlat * nlon - len(raster_data)), "constant"
                    )
                global_surf_read.raster = raster_data.reshape((nlat, nlon)).T

                first_lat = global_surf_read.lati[0]
                last_lat = global_surf_read.lati[nlat - 1]
                global_surf_read.max_lat = max(first_lat, last_lat)
                global_surf_read.min_lat = min(first_lat, last_lat)

                first_lon = global_surf_read.loni[0]
                last_lon = global_surf_read.loni[nlon - 1]
                global_surf_read.max_lon = max(first_lon, last_lon)
                global_surf_read.min_lon = min(first_lon, last_lon)

                return global_surf_read

        except FileNotFoundError:
            self.log(f"Error surface file {surface_file} not found.")
            exit(1)
        except Exception as e:
            self.log(f"Error: {e}")
            exit(1)

    def load_global_surface_data(self, global_surface_names: List[str]):
        """
        Load all global surface data.

        Parameters
        ----------
        global_surface_names : List[str]
            List of global surface names to load.

        Returns
        -------
        GlobalSurfaces
            The loaded global surfaces.
        """
        surfaces = [self.get_info("surface", name) for name in global_surface_names]
        global_surfaces = GlobalSurfaces()

        for surface in surfaces:
            global_surfaces.surface.append(self.load_global_surface(surface["path"]))

        return global_surfaces

    def load_smooth_boundaries(
        self, nz_tomography_data: TomographyData, basin_names: List[str]
    ):
        """
        Load smooth boundaries for the tomography data.

        Parameters
        ----------
        nz_tomography_data : TomographyData
            The tomography data to load smooth boundaries for.
        basin_names : List[str]
            List of basin names to load smooth boundaries for.
        """
        smooth_bound = nz_tomography_data.smooth_boundary
        count = 0

        for basin_name in basin_names:
            basin = self.get_info("basin", basin_name)
            if basin is None:
                self.log(f"Error: Basin {basin_name} not found in registry.")
                exit(1)
            if "smoothing" in basin:
                # Assumed a single smoothing file defined for a basin
                boundary_vec_filename = self.get_full_path(basin["smoothing"])

                if boundary_vec_filename.exists():
                    self.log(
                        f"Loading offshore smoothing file: {boundary_vec_filename}."
                    )
                    try:
                        data = np.fromfile(boundary_vec_filename, dtype=float, sep=" ")
                        x_pts = data[0::2]
                        y_pts = data[1::2]
                        smooth_bound.xpts.extend(x_pts)
                        smooth_bound.ypts.extend(y_pts)
                        count += len(x_pts)
                    except Exception as e:
                        self.log(
                            f"Error reading smoothing boundary vector file {boundary_vec_filename}: {e}"
                        )
                else:
                    self.log(
                        f"Error smoothing boundary vector file {boundary_vec_filename} not found."
                    )
            else:
                self.log(f"Smoothing not required for basin {basin_name}.")
        smooth_bound.n = count
