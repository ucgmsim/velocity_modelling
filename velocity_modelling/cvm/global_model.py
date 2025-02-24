import numpy as np

from velocity_modelling.cvm.interpolate import (
    bi_linear_interpolation,
    linear_interpolation,
)
from velocity_modelling.cvm.geometry import (
    AdjacentPoints,
    MeshVector,
)


class GlobalSurfaces:
    def __init__(self):
        """
        Initialize the GlobalSurfaces.
        """
        self.surface = []


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
            n_velo_ind = np.where(self.depth >= depth)[0][-1]
            if n_velo_ind == len(self.depth):
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
        for i in range(len(global_surfaces.surface)):
            global_surf_read = global_surfaces.surface[i]
            adjacent_points = AdjacentPoints.find_global_adjacent_points(
                global_surf_read.lati,
                global_surf_read.loni,
                mesh_vector.lat,
                mesh_vector.lon,
            )
            self.depth[i] = interpolate_global_surface(
                global_surf_read, mesh_vector, adjacent_points
            )

        depths = self.depth

        # Find indices where top_val < bot_val
        mask = depths[:-1] < depths[1:]

        # Apply the condition using NumPy indexing
        depths[1:][mask] = depths[:-1][mask]


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
