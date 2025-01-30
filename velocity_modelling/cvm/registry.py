import yaml

from enum import Enum
import numpy as np
from typing import List, Dict
from pathlib import Path

from logging import Logger
import logging
import sys

from qcore import point_in_polygon
from qcore import coordinates

from velocity_modelling.cvm.constants import MAX_LAT_SURFACE_EXTENSION, MAX_LON_SURFACE_EXTENSION

DATA_ROOT = Path(__file__).parent.parent / "Data"
nzvm_registry_path = DATA_ROOT / "nzvm_registry.yaml"

DEFAULT_OFFSHORE_1D_MODEL = "Cant1D_v2"  # vm1d name for offshore 1D model
DEFAULT_OFFSHORE_DISTANCE = "offshore"  # surface name for offshore distance


class CVMRegistry:  # Forward declaration
    pass


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
    def __init__(self, nLat: int, nLon: int):
        """
        Initialize the BasinSurfaceRead.

        Parameters
        ----------
        nLat : int
            The number of latitude points.
        nLon : int
            The number of longitude points.
        """
        self.nLat = nLat
        self.nLon = nLon
        self.lati = np.zeros(nLat)
        self.loni = np.zeros(nLon)
        self.raster = np.zeros((nLon, nLat))
        self.maxLat = None
        self.minLat = None
        self.maxLon = None
        self.minLon = None


class GlobalMesh:
    def __init__(self, nX: int, nY: int, nZ: int):
        """
        Initialize the GlobalMesh.

        Parameters
        ----------
        nX : int
            The number of points in the X direction.
        nY : int
            The number of points in the Y direction.
        nZ : int
            The number of points in the Z direction.
        """
        self.Lon = np.zeros((nX, nY))
        self.Lat = np.zeros((nX, nY))
        self.maxLat = 0.0
        self.minLat = 0.0
        self.maxLon = 0.0
        self.minLon = 0.0
        self.nX = nX
        self.nY = nY
        self.nZ = nZ
        self.X = np.zeros(nX)
        self.Y = np.zeros(nY)
        self.Z = np.zeros(nZ)


class PartialGlobalSurfaceDepths:
    def __init__(self, nSurfDep: int):
        """
        Initialize the PartialGlobalSurfaceDepth.

        Parameters
        ----------
        nSurfDep : int
            The number of surface depths.
        """
        self.dep = np.zeros(nSurfDep, dtype=np.float64)
        self.nSurfDep = nSurfDep

    def interpolate_global_surface_depths(
        self, global_surfaces, mesh_vector, calculation_log
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
        for i in range(global_surfaces.nSurf):
            global_surf_read = global_surfaces.surf[i]
            adjacent_points = AdjacentPoints()
            adjacent_points = global_surf_read.find_global_adjacent_points(mesh_vector)
            self.dep[i] = global_surf_read.interpolate_global_surface(
                mesh_vector, adjacent_points
            )

        self.nSurfDep = global_surfaces.nSurf
        for i in range(global_surfaces.nSurf - 1, 0, -1):
            top_val = self.dep[i - 1]
            bot_val = self.dep[i]
            if top_val < bot_val:
                self.dep[i] = top_val
                # calculation_log.nPointsGlobalSurfacesEnforced += 1


class PartialBasinSurfaceDepths:
    def __init__(self, num_basins: int, num_basin_surfaces: int):
        self.dep = np.zeros((num_basins, num_basin_surfaces), dtype=np.float64)


class PartialGlobalMesh:
    def __init__(self, nX: int, nZ: int):
        """
        Initialize the PartialGlobalMesh.

        Parameters
        ----------
        nX : int
            The number of points in the X direction.
        nZ : int
            The number of points in the Z direction.
        """
        self.Lon = np.zeros(nX)
        self.Lat = np.zeros(nX)
        self.X = np.zeros(nX)
        self.Z = np.zeros(nZ)
        self.nX = nX
        self.nY = 1
        self.nZ = nZ
        self.Y = 0.0


class MeshVector:
    def __init__(self, nZ, lat=None, lon=None):
        self.Lat = lat
        self.Lon = lon
        self.Z = np.zeros(nZ)
        self.nZ = nZ
        self.Vs30 = None
        self.distance_from_shoreline = None


class PartialGlobalQualities:
    def __init__(self, lon_grid_dim_max: int, dep_grid_dim_max: int):
        self.Vp = np.zeros((lon_grid_dim_max, dep_grid_dim_max), dtype=np.float64)
        self.Vs = np.zeros((lon_grid_dim_max, dep_grid_dim_max), dtype=np.float64)
        self.Rho = np.zeros((lon_grid_dim_max, dep_grid_dim_max), dtype=np.float64)
        self.inbasin = np.zeros((lon_grid_dim_max, dep_grid_dim_max), dtype=bool)


class GlobalSurfaces:
    def __init__(self):
        """
        Initialize the GlobalSurfaces.
        """
        self.surf = []


class GlobalSurfaceRead:
    def __init__(self, nLat: int, nLon: int):
        """
        Initialize the GlobalSurfaceRead.

        Parameters
        ----------
        nLat : int
            The number of latitude points.
        nLon : int
            The number of longitude points.
        """
        self.nLat = nLat
        self.nLon = nLon
        self.lati = np.zeros(nLat)
        self.loni = np.zeros(nLon)
        self.raster = np.zeros((nLon, nLat))
        self.maxLat = None
        self.minLat = None
        self.maxLon = None
        self.minLon = None

    def find_global_adjacent_points(self, mesh_vector: MeshVector):
        """
        Find the adjacent points to the mesh vector in the global surface.

        Parameters
        ----------
        mesh_vector : MeshVector
            The mesh vector.

        Returns
        -------
        Tuple[int, int, int, int]
            The indices of the adjacent points.
        """

        import bisect

        lat = mesh_vector.Lat
        lon = mesh_vector.Lon

        lat_assigned_flag = False
        lon_assigned_flag = False

        adjacent_points = AdjacentPoints()
        adjacent_points.in_surface_bounds = False
        adjacent_points.in_lat_extension_zone = False
        adjacent_points.in_lon_extension_zone = False
        adjacent_points.in_corner_zone = False

        # Binary search for latitude
        lat_index = bisect.bisect_left(self.lati, lat)
        if lat_index < len(self.lati) and self.lati[lat_index] == lat:
            adjacent_points.lat_ind[0] = lat_index
            adjacent_points.lat_ind[1] = lat_index
            lat_assigned_flag = True
        elif 0 < lat_index < len(self.lati):
            adjacent_points.lat_ind[0] = lat_index - 1
            adjacent_points.lat_ind[1] = lat_index
            lat_assigned_flag = True

        # Binary search for longitude
        lon_index = bisect.bisect_left(self.loni, lon)
        if lon_index < len(self.loni) and self.loni[lon_index] == lon:
            adjacent_points.lon_ind[0] = lon_index
            adjacent_points.lon_ind[1] = lon_index
            lon_assigned_flag = True
        elif 0 < lon_index < len(self.loni):
            adjacent_points.lon_ind[0] = lon_index - 1
            adjacent_points.lon_ind[1] = lon_index
            lon_assigned_flag = True

        if not lat_assigned_flag or not lon_assigned_flag:
            if lon_assigned_flag and not lat_assigned_flag:
                if (
                    lat - self.maxLat
                ) <= MAX_LAT_SURFACE_EXTENSION and lat >= self.maxLat:
                    adjacent_points.in_lat_extension_zone = True
                    self.find_edge_inds(adjacent_points, 1)
                elif (
                    self.minLat - lat
                ) <= MAX_LAT_SURFACE_EXTENSION and lat <= self.minLat:
                    adjacent_points.in_lat_extension_zone = True
                    self.find_edge_inds(adjacent_points, 3)

            if lat_assigned_flag and not lon_assigned_flag:
                if (
                    self.minLon - lon
                ) <= MAX_LON_SURFACE_EXTENSION and lon <= self.minLon:
                    adjacent_points.in_lon_extension_zone = True
                    self.find_edge_inds(adjacent_points, 4)
                elif (
                    lon - self.maxLon
                ) <= MAX_LON_SURFACE_EXTENSION and lon >= self.maxLon:
                    adjacent_points.in_lon_extension_zone = True
                    self.find_edge_inds(adjacent_points, 2)

            if (
                (lat - self.maxLat) <= MAX_LAT_SURFACE_EXTENSION
                and (self.minLon - lon) <= MAX_LON_SURFACE_EXTENSION
                and lon <= self.minLon
                and lat >= self.maxLat
            ):
                self.find_corner_inds(self.maxLat, self.minLon, adjacent_points)
            elif (
                (lat - self.maxLat) <= MAX_LAT_SURFACE_EXTENSION
                and (lon - self.maxLon) <= MAX_LON_SURFACE_EXTENSION
                and lon >= self.maxLon
                and lat >= self.maxLat
            ):
                self.find_corner_inds(self.maxLat, self.maxLon, adjacent_points)
            elif (
                (self.minLat - lat) <= MAX_LAT_SURFACE_EXTENSION
                and (self.minLon - lon) <= MAX_LON_SURFACE_EXTENSION
                and lon <= self.minLon
                and lat <= self.minLat
            ):
                self.find_corner_inds(self.minLat, self.minLon, adjacent_points)
            elif (
                (self.minLat - lat) <= MAX_LAT_SURFACE_EXTENSION
                and (lon - self.maxLon) <= MAX_LON_SURFACE_EXTENSION
                and lon >= self.maxLon
                and lat <= self.minLat
            ):
                self.find_corner_inds(self.minLat, self.maxLon, adjacent_points)

            if (
                not adjacent_points.in_lat_extension_zone
                and not adjacent_points.in_lon_extension_zone
                and not adjacent_points.in_corner_zone
            ):
                raise ValueError("Point does not lie in any global surface extension.")
        else:
            adjacent_points.in_surface_bounds = True

        return adjacent_points

    def find_edge_inds(self, adjacent_points, edge_type):
        """
        Find the indices of the edge of the global surface closest to the lat-lon point.

        Parameters:
        adjacent_points (AdjacentPoints): Object containing indices of points adjacent to the lat-lon for interpolation.
        edge_type (int): Indicating whether the point lies to the north, east, south, or west of the global surface.
        """
        if edge_type == 1:
            if self.maxLat == self.lati[0]:
                adjacent_points.lat_edge_ind = 0
            elif self.maxLat == self.lati[self.nLat - 1]:
                adjacent_points.lat_edge_ind = self.nLat - 1
            else:
                raise ValueError("Point lies outside of surface bounds.")
        elif edge_type == 3:
            if self.minLat == self.lati[0]:
                adjacent_points.lat_edge_ind = 0
            elif self.minLat == self.lati[self.nLat - 1]:
                adjacent_points.lat_edge_ind = self.nLat - 1
            else:
                raise ValueError("Point lies outside of surface bounds.")
        elif edge_type == 2:
            if self.maxLon == self.loni[0]:
                adjacent_points.lon_edge_ind = 0
            elif self.maxLon == self.loni[self.nLon - 1]:
                adjacent_points.lon_edge_ind = self.nLon - 1
            else:
                raise ValueError("Point lies outside of surface bounds.")
        elif edge_type == 4:
            if self.minLon == self.loni[0]:
                adjacent_points.lon_edge_ind = 0
            elif self.minLon == self.loni[self.nLon - 1]:
                adjacent_points.lon_edge_ind = self.nLon - 1
            else:
                raise ValueError("Point lies outside of surface bounds.")
        else:
            raise ValueError("Point lies outside of surface bounds.")

    def find_corner_inds(self, lat_pt, lon_pt, adjacent_points):
        """
        Find the indices of the corner of the global surface closest to the lat-lon point.

        Parameters:
        lat_pt (float): Latitude of point for eventual interpolation.
        lon_pt (float): Longitude of point for eventual interpolation.
        adjacent_points (AdjacentPoints): Object containing indices of points adjacent to the lat-lon for interpolation.
        """
        if lat_pt == self.lati[0]:
            adjacent_points.corner_lat_ind = 0
        elif lat_pt == self.lati[self.nLat - 1]:
            adjacent_points.corner_lat_ind = self.nLat - 1
        else:
            raise ValueError("Point lies outside of surface bounds.")

        if lon_pt == self.loni[0]:
            adjacent_points.corner_lon_ind = 0
        elif lon_pt == self.loni[self.nLon - 1]:
            adjacent_points.corner_lon_ind = self.nLon - 1
        else:
            raise ValueError("Point lies outside of surface bounds.")

        adjacent_points.in_corner_zone = 1

    def interpolate_global_surface(self, mesh_vector: MeshVector, adjacent_points):
        """
        Interpolate the global surface value at a given latitude and longitude.

        Parameters:
        lat (float): Latitude of the point for interpolation.
        lon (float): Longitude of the point for interpolation.
        adjacent_points (AdjacentPoints): Object containing indices of points adjacent to the lat-lon for interpolation.

        Returns:
        float: Interpolated value at the given lat-lon.
        """
        lat = mesh_vector.Lat
        lon = mesh_vector.Lon

        lat1 = self.lati[adjacent_points.lat_ind[0]]
        lat2 = self.lati[adjacent_points.lat_ind[1]]
        lon1 = self.loni[adjacent_points.lon_ind[0]]
        lon2 = self.loni[adjacent_points.lon_ind[1]]

        f11 = self.raster[adjacent_points.lat_ind[0]][adjacent_points.lon_ind[0]]
        f12 = self.raster[adjacent_points.lat_ind[0]][adjacent_points.lon_ind[1]]
        f21 = self.raster[adjacent_points.lat_ind[1]][adjacent_points.lon_ind[0]]
        f22 = self.raster[adjacent_points.lat_ind[1]][adjacent_points.lon_ind[1]]

        # bilinear interpolation between the four points
        interpolated_value = (
            f11 * (lat2 - lat) * (lon2 - lon)
            + f21 * (lat - lat1) * (lon2 - lon)
            + f12 * (lat2 - lat) * (lon - lon1)
            + f22 * (lat - lat1) * (lon - lon1)
        ) / ((lat2 - lat1) * (lon2 - lon1))

        return interpolated_value


class ModelExtent:
    def __init__(self, vm_params: Dict):
        """
        Initialize the ModelExtent.

        Parameters
        ----------
        vm_params : Dict
            The velocity model parameters.
        """
        self.originLat = vm_params["MODEL_LAT"]
        self.originLon = vm_params["MODEL_LON"]
        self.originRot = vm_params["MODEL_ROT"]  # in degrees
        self.Xmax = vm_params["extent_x"]
        self.Ymax = vm_params["extent_y"]
        self.Zmax = vm_params["extent_zmax"]
        self.Zmin = vm_params["extent_zmin"]
        self.hDep = vm_params["hh"]
        self.hLatLon = vm_params["hh"]
        self.nx = vm_params["nx"]
        self.ny = vm_params["ny"]
        self.nz = vm_params["nz"]


class SmoothingBoundary:
    def __init__(self):
        """
        Initialize the SmoothingBoundary.
        """
        self.n = 0
        self.xPts = []
        self.yPts = []

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

        return closest_ind, distance

    def brute_force(self, mesh_vector: MeshVector):
        """
        Determine the closest index within the smoothing boundary to the given mesh vector coordinates.

        Parameters:
        smoothing_boundary (SmoothingBoundary): Object containing smoothing boundary data.
        mesh_vector (MeshVector): Object containing mesh vector data.

        Returns:
        int: Index of the closest point in the smoothing boundary.
        """
        boundary_points = np.column_stack((self.yPts, self.xPts))
        mesh_point = np.array([mesh_vector.Lat, mesh_vector.Lon])

        distances = coordinates.distance_between_wgs_depth_coordinates(
            boundary_points, mesh_point
        )
        closest_ind = np.argmin(distances)
        return closest_ind, distances[closest_ind]


class VeloMod1DData:
    def __init__(self):
        """
        Initialize the VeloMod1DData.
        """
        self.Vp = []
        self.Vs = []
        self.Rho = []
        self.Dep = []
        self.nDep = 0


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

        self.surfDeps = tomo["elev"]
        self.surf = []

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

        for i in range(len(self.surfDeps)):
            self.surf.append({})  # self.surf[i] is an empty dictionary
            elev = self.surfDeps[i]
            if elev == int(elev):  # if the elevation is an integer
                elev_name = f"{elev}"
            else:
                elev_name = f"{elev:.2f}".replace(".", "p")

            for vtype in VTYPE:
                tomofile = (
                    surf_tomo_path / f"surf_tomography_{vtype.name}_elev{elev_name}.in"
                )
                self.surf[i][vtype.name] = cvm_registry.load_global_surface(tomofile)

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

        mesh_vector.Vs30 = self.vs30.interpolate_global_surface(
            mesh_vector, adjacent_points
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
        mesh_vector.distance_from_shoreline = (
            self.offshore_distance_surface.interpolate_global_surface(
                mesh_vector, adjacent_points
            )
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


class InBasin:
    def __init__(self, num_basins: int, num_basin_boundaries: int, dep_grid_dim: int):
        self.inBasinLatLon = np.zeros((num_basins, num_basin_boundaries), dtype=bool)
        self.inBasinDep = np.zeros((num_basins, dep_grid_dim), dtype=bool)


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

        self.boundaries = [] # List of basin boundaries
        basin_info = cvm_registry.get_info("basin", basin_name)

        self.surf = [
            cvm_registry.load_basin_surface(surface["path"])
            for surface in basin_info["surfaces"]
        ]
        self.boundary = [
            cvm_registry.load_basin_boundary(boundary["path"])
            for boundary in basin_info["boundaries"]
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

    def point_on_vertex(self, i: int, mesh_vector: MeshVector) -> bool:
        """
        Check if a point lies on a vertex of a basin boundary.

        Parameters
        ----------
        i : int
            The index of the boundary.
        mesh_vector : MeshVector
            The mesh vector containing the point.

        Returns
        -------
        bool
            True if the point lies on a vertex of the boundary, False otherwise.
        """
        boundary_lats = self.boundary_lat(i)
        boundary_lons = self.boundary_lon(i)
        on_vertex = np.any(np.isclose(boundary_lats, mesh_vector.Lat) & np.isclose(boundary_lons,  mesh_vector.Lon))
        return on_vertex


    def determine_if_within_basin_lat_lon(self, mesh_vector: MeshVector, basin_num: int = -1, in_basin: InBasin = None):
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
        if in_basin:
            assert basin_num >= 0, "Basin number must be provided if in_basin is provided."

        for j in range(len(self.boundary)):
            boundary = self.boundary[j]

            if not (
                np.min(boundary[:, 0]) <= mesh_vector.lon <= np.max(boundary[:, 0])
                and np.min(boundary[:, 1]) <= mesh_vector.lat <= np.max(boundary[:, 1])
            ):
                continue  # outside of basin

            else:
                # possibly in basin
                in_poly = point_in_polygon.is_inside_postgis(
                    self.boundary[j], np.array([mesh_vector.lon, mesh_vector.lat])
                )  # check if in poly

                if in_poly:
                    if in_basin:
                        in_basin.inBasinLatLon[basin_num][j] = True
                    return True  # inside a basin (any)
                else: # outside poly
                    if in_basin: #check if it is on vertex
                        in_basin.inBasinLatLon[basin_num][j] = self.point_on_vertex(j, mesh_vector)

                    continue  # outside of basin

        return False  # not inside basin

    # def determine_basin_surface_depths(self, basin_num: int, in_basin: InBasin, partial_basin_surface_depths: PartialBasinSurfaceDepths,
    #                                    mesh_vector: MeshVector):
    #     """
    #     Determine the basin surface depths for a given latitude and longitude.
    #
    #     Parameters
    #     ----------
    #     in_basin : InBasin
    #         Struct containing flags to indicate if lat-lon point - depths lie within the basin.
    #     partial_basin_surface_depths : PartialBasinSurfaceDepths
    #         Struct containing depths for all applicable basin surfaces at one lat-lon location.
    #     lat : float
    #         The latitude of the point.
    #     lon : float
    #         The longitude of the point.
    #     """
    #     for j in range(len(self.surf)):
    #         boundary_num =
    #         if in_basin.inBasinLatLon[basin_num][j]:
    #             adjacent_points = self.surf[j].find_global_adjacent_points(mesh_vector)
    #             partial_basin_surface_depths.dep[basin_num][j] = self.surf[j].interpolate_global_surface(mesh_vector,adjacent_points)

    # def interpolate_basin_surface_depths(self, basin_num: int, in_basin: InBasin, partial_basin_surface_depths: PartialBasinSurfaceDepths, mesh_vector: MeshVector):
    #     """
    #     Determine if a lat-lon point is in a basin, if so interpolate the basin surface depths, enforce their hierarchy, then determine which depth points lie within the basin limits.
    #
    #     Parameters
    #     ----------
    #     in_basin : InBasin
    #         Struct containing flags to indicate if lat-lon point - depths lie within the basin.
    #     partial_basin_surface_depths : PartialBasinSurfaceDepths
    #         Struct containing depths for all applicable basin surfaces at one lat-lon location.
    #     mesh_vector : MeshVector
    #         Struct containing a single lat-lon point with one or more depths.
    #     """
    #     self.determine_if_within_basin_lat_lon(mesh_vector, basin_num, in_basin)
    #     self.determine_basin_surface_depths(in_basin, partial_basin_surface_depths, mesh_vector.Lat, mesh_vector.Lon)
    #     self.enforce_basin_surface_depths(in_basin, partial_basin_surface_depths, mesh_vector)

class QualitiesVector:
    def __init__(self, dep_grid_dim_max: int):
        self.Vp = np.zeros(dep_grid_dim_max, dtype=np.float64)
        self.Vs = np.zeros(dep_grid_dim_max, dtype=np.float64)
        self.Rho = np.zeros(dep_grid_dim_max, dtype=np.float64)
        self.inbasin = np.zeros(dep_grid_dim_max, dtype=bool)

    def prescribe_velocities(
        self,
        velo_mod_1d_data: VeloMod1DData,
        nz_tomography_data: TomographyData,
        global_surfaces: GlobalSurfaces,
        basin_data_list: List[BasinData],
        mesh_vector: MeshVector,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        in_basin: InBasin,
        topo_type: str,
        on_boundary: bool,
        calculation_log: Logger,
    ):

        partial_global_surface_depths.interpolate_global_surface_depths(
            global_surfaces, mesh_vector, calculation_log
        )

        dZ = 0
        depth_change = 0

        shifted_mesh_vector = None

        in_any_basin_lat_lon = np.any(
            [
                basin_data.determine_if_within_basin_lat_lon(mesh_vector)
                for basin_data in basin_data_list
            ]
        )

        if topo_type == "SQUASHED":
            shifted_mesh_vector = MeshVector(mesh_vector.nZ, lat=mesh_vector.Lat, lon=mesh_vector.Lon)

            depth_change = -mesh_vector.Z  # is this correct????
            shifted_mesh_vector.Z = partial_global_surface_depths.dep[1] - depth_change

        elif topo_type == "SQUASHED_TAPERED":
            dZ = mesh_vector.Z[0] - mesh_vector.Z[1]
            TAPER_DIST = 1.0
            shifted_mesh_vector = MeshVector(mesh_vector.nZ, lat=mesh_vector.Lat, lon=mesh_vector.Lon)

            depth_change = -mesh_vector.Z
            TAPER_VAL = np.where(
                (depth_change == 0) | (partial_global_surface_depths.dep[1] == 0) | (
                            partial_global_surface_depths.dep[1] < 0),
                1.0,
                1.0 - (depth_change / (partial_global_surface_depths.dep[1] * TAPER_DIST))
            )
            TAPER_VAL = np.clip(TAPER_VAL, 0.0, None)
            shifted_mesh_vector.Z = partial_global_surface_depths.dep[1] * TAPER_VAL - depth_change

        elif topo_type in ["BULLDOZED", "TRUE"]:
            shifted_mesh_vector = mesh_vector

        else:
            raise ValueError("User specified TOPO_TYPE not recognised, see readme.")

        for basin_num, basin_data in enumerate(basin_data_list):
            basin_data.interpolate_basin_surface_depths(basin_num,
                in_basin, partial_basin_surface_depths, shifted_mesh_vector
            )

        basin_flag = 0
        Z = 0
        for k in range(mesh_vector.nz):
            if topo_type in ["BULLDOZED", "TRUE"]:
                Z = mesh_vector.z[k]
            elif topo_type in ["SQUASHED", "SQUASHED_TAPERED"]:
                Z = shifted_mesh_vector.z[k]

            for i in range(len(basin_data_list)):
                if in_basin.in_basin_dep[i][k] == 1:
                    basin_flag = 1
                    self.inbasin[k] = True
                    basin_data = basin_data_list[i]
                    basin_data.assign_basin_qualities(
                        partial_basin_surface_depths,
                        partial_global_surface_depths,
                        nz_tomography_data,
                        mesh_vector,
                        in_any_basin_lat_lon,
                        on_boundary,
                        Z,
                        i,
                        k,
                    )

            if basin_flag == 0:
                self.inbasin[k] = False
                n_velo_mod_ind = self.find_global_sub_velo_model_ind(
                    Z, partial_global_surface_depths
                )
                self.call_sub_velocity_model(
                    n_velo_mod_ind,
                    k,
                    Z,
                    mesh_vector,
                    nz_tomography_data,
                    partial_global_surface_depths,
                    in_any_basin_lat_lon,
                    on_boundary,
                )

            if Z > partial_global_surface_depths.dep[1]:
                self.nan_sub_mod(k)

            basin_flag = 0

        if topo_type == "BULLDOZED":
            for k in range(mesh_vector.nz):
                if mesh_vector.z[k] > 0:
                    self.nan_sub_mod(k)

        if shifted_mesh_vector:
            del shifted_mesh_vector


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
        velo_mod_1d_data = VeloMod1DData()
        try:
            with open(v1d_path, "r") as file:
                next(file)
                data = np.loadtxt(file)
                velo_mod_1d_data.Vp = data[:, 0].tolist()
                velo_mod_1d_data.Vs = data[:, 1].tolist()
                velo_mod_1d_data.Rho = data[:, 2].tolist()
                velo_mod_1d_data.Dep = data[:, 5].tolist()
                velo_mod_1d_data.nDep = len(velo_mod_1d_data.Dep)
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
        basin_surface : dict
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
            return self.load_1d_velo_sub_model(vm1d["path"])

        elif submodel["type"] == "relation":
            return VeloMod1DData()
        elif submodel["type"] == "perturbation":
            return VeloMod1DData()

    def load_basin_surface(self, basin_surface_path: Path):
        """
        Load a basin surface from a file.

        Parameters
        ----------
        basin_surface_path : Path
            The path to the basin surface file.

        Returns
        -------
        BasinSurfaceRead
            The loaded basin surface data.
        """
        self.log(f"Loading basin surface file {basin_surface_path}")

        basin_surface_path = self.get_full_path(basin_surface_path)

        try:
            with open(basin_surface_path, "r") as f:
                nLat, nLon = map(int, f.readline().split())
                basin_surf_read = BasinSurfaceRead(nLat, nLon)

                latitudes = np.fromfile(f, dtype=float, count=nLat, sep=" ")
                longitudes = np.fromfile(f, dtype=float, count=nLon, sep=" ")

                basin_surf_read.lati = latitudes
                basin_surf_read.loni = longitudes

                raster_data = np.fromfile(f, dtype=float, count=nLat * nLon, sep=" ")
                assert (
                    len(raster_data) == nLat * nLon
                ), f"Error: in {basin_surface_path} raster data length mismatch: {len(raster_data)} != {nLat * nLon}"
                basin_surf_read.raster = raster_data.reshape((nLon, nLat)).T

                firstLat = basin_surf_read.lati[0]
                lastLat = basin_surf_read.lati[nLat - 1]
                basin_surf_read.maxLat = max(firstLat, lastLat)
                basin_surf_read.minLat = min(firstLat, lastLat)

                firstLon = basin_surf_read.loni[0]
                lastLon = basin_surf_read.loni[nLon - 1]
                basin_surf_read.maxLon = max(firstLon, lastLon)
                basin_surf_read.minLon = min(firstLon, lastLon)

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
        for i in range(len(global_model_params["velo_submodels"])):
            if global_model_params["velo_submodels"][i] == "v1DsubMod":
                velo_mod_1d_data = self.load_1d_velo_sub_model(
                    global_model_params["velo_submodels"][i]
                )
                self.log("Loaded 1D velocity model data.")
            elif global_model_params["velo_submodels"][i] == "NaNsubMod":
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
                nLat, nLon = map(int, f.readline().split())
                global_surf_read = GlobalSurfaceRead(nLat, nLon)

                latitudes = np.fromfile(f, dtype=float, count=nLat, sep=" ")
                longitudes = np.fromfile(f, dtype=float, count=nLon, sep=" ")

                global_surf_read.lati = latitudes
                global_surf_read.loni = longitudes

                raster_data = np.fromfile(f, dtype=float, count=nLat * nLon, sep=" ")
                try:
                    global_surf_read.raster = raster_data.reshape((nLon, nLat)).T
                except:
                    self.log(
                        f"Error: in {surface_file} raster data length mismatch: {len(raster_data)} != {nLat * nLon}"
                    )
                    exit(1)

                firstLat = global_surf_read.lati[0]
                lastLat = global_surf_read.lati[nLat - 1]
                global_surf_read.maxLat = max(firstLat, lastLat)
                global_surf_read.minLat = min(firstLat, lastLat)

                firstLon = global_surf_read.loni[0]
                lastLon = global_surf_read.loni[nLon - 1]
                global_surf_read.maxLon = max(firstLon, lastLon)
                global_surf_read.minLon = min(firstLon, lastLon)

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
            global_surfaces.surf.append(self.load_global_surface(surface["path"]))

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
            if "smoothing" in basin:
                boundary_vec_filename = self.get_full_path(basin["smoothing"]["path"])

                if boundary_vec_filename.exists():
                    self.log(
                        f"Loading offshore smoothing file: {boundary_vec_filename}."
                    )
                    try:
                        data = np.fromfile(boundary_vec_filename, dtype=float, sep=" ")
                        x_pts = data[0::2]
                        y_pts = data[1::2]
                        smooth_bound.xPts.extend(x_pts)
                        smooth_bound.yPts.extend(y_pts)
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
