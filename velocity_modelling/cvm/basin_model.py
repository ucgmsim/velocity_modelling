import numpy as np
from logging import Logger
import logging

import sys

from velocity_modelling.cvm.registry import CVMRegistry
from velocity_modelling.cvm.interpolate import bi_linear_interpolation
from velocity_modelling.cvm.geometry import point_on_vertex, AdjacentPoints, MeshVector
from qcore import point_in_polygon


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

    def determine_if_within_basin_lat_lon(self, mesh_vector: MeshVector):
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

        for ind, boundary in enumerate(self.boundaries):

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
                    return True  # inside a basin (any)

        return False  # not inside basin


class InBasin:
    def __init__(self, basin_data: BasinData, n_depths: int):
        """
        Initialize the InBasin.

        Parameters
        ----------
        basin_data : BasinData
            The BasinData instance.
        n_depths : int
            The number of depth points.
        """
        self.basin_data = basin_data
        self.in_basin_lat_lon = np.zeros(len(basin_data.boundaries), dtype=bool)
        self.in_basin_depth = np.zeros((n_depths), dtype=bool)

    def determine_if_within_basin_lat_lon(self, mesh_vector: MeshVector):
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

        for ind, boundary in enumerate(self.basin_data.boundaries):

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
                    self.in_basin_lat_lon[ind] = True
                    return True  # inside a basin (any)
                else:  # outside poly
                    # check if it is on vertex. if in_poly
                    on_vertex = point_on_vertex(
                        self.basin_data.boundary_lat(ind),
                        self.basin_data.boundary_lon(ind),
                        mesh_vector.lat,
                        mesh_vector.lon,
                    )
                    if on_vertex:
                        self.in_basin_lat_lon[ind] = True
                        return True
                    continue  # outside of basin

        return False  # not inside basin


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
        self.depths = np.zeros(len(basin_data.surfaces), dtype=np.float64)
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
        for surface_ind, surface in enumerate(inbasin.basin_data.surfaces):

            if np.any(
                inbasin.in_basin_lat_lon
            ):  # see if this is in any boundary of this basin
                adjacent_points = AdjacentPoints.find_basin_adjacent_points(
                    surface.lati, surface.loni, mesh_vector.lat, mesh_vector.lon
                )
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
            else:
                self.depths[surface_ind] = None

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

    def determine_basin_surface_above(self, depth):
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

        if np.any(in_basin.in_basin_lat_lon):

            self.enforce_surface_depths()
            # TODO: check if this is correct
            top_lim = self.depths[0]  # the depth of the first surface
            bot_lim = self.depths[-1]  # the depth of the last surface

            in_basin.in_basin_depth = (bot_lim <= mesh_vector.z) & (
                mesh_vector.z <= top_lim
            )  # check if the point is within the basin
        else:
            in_basin.in_basin_depth.fill(False)

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
        in_basin.determine_if_within_basin_lat_lon(mesh_vector)
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
