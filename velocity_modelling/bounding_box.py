"""
Module for handling bounding boxes in 2D space.

This module provides classes and functions for working with bounding boxes in
2D space, including calculating axis-aligned and minimum area bounding boxes,
and computing various properties such as area and bearing. The bounding box
dimensions are in metres except where otherwise mentioned.

Classes
-------
- BoundingBox: Represents a 2D bounding box with properties and methods for calculations.

Functions
---------
- axis_aligned_bounding_box: Returns an axis-aligned bounding box containing points.
- rotation_matrix: Returns the 2D rotation matrix for a given angle.
- minimum_area_bounding_box: Returns the smallest rectangle bounding points.
- minimum_area_bounding_box_for_polygons_masked: Returns a bounding box around masked polygons.

References
----------
- BoundingBox wiki page: https://github.com/ucgmsim/qcore/wiki/BoundingBox
"""

from typing import Self

import numpy as np
import numpy.typing as npt
import shapely
from shapely import Polygon

from qcore import coordinates, geo


class BoundingBox:
    """Represents a 2D bounding box with properties and methods for calculations.

    Parameters
    ----------
    bounds : npt.NDArray[np.float64]
        The bounds of the box in NZTM coordinates.

    Attributes
    ----------
    corners : np.ndarray
       The corners of the bounding box in cartesian coordinates. The
       order of the corners should be counter clock-wise from the bottom-left point
       (minimum x, minimum y).
    """

    bounds: npt.NDArray[np.float64]

    def __init__(self, bounds: npt.NDArray[np.float64]):
        """Create a bounding box from bounds in NZTM coordinates.

        Parameters
        ----------
        bounds : npt.NDArray[np.float64]
            The bounds of the box.
        """
        bottom_left_index = (bounds - np.mean(bounds, axis=0)).sum(axis=1).argmin()
        bounds = np.copy(bounds)
        bounds[[0, bottom_left_index]] = bounds[[bottom_left_index, 0]]
        angles = np.arctan2(*(bounds[1:] - bounds[0]).T)

        indices = np.argsort(angles, kind="stable") + 1
        self.bounds = np.vstack([bounds[0], bounds[indices]])

    @property
    def corners(self) -> np.ndarray:  # numpydoc ignore=RT01
        """np.ndarray: the corners of the bounding box in (lat, lon) format."""
        return coordinates.nztm_to_wgs_depth(self.bounds)

    def pad(
        self,
        pad_x: tuple[float, float] = (0.0, 0.0),
        pad_y: tuple[float, float] = (0.0, 0.0),
    ) -> Self:
        """Pad the bounding box by extending it in the x and y directions.

        Parameters
        ----------
        pad_x : tuple[float, float], default (0, 0)
            Padding distances in kilometers for x direction (left, right).
        pad_y : tuple[float, float], default (0, 0)
            Padding distances in kilometers for y direction (bottom, top).

        Returns
        -------
        Self
            A new instance of the bounding box with applied padding
        """
        bounds = self.bounds
        x_direction = bounds[1] - bounds[0]
        x_direction /= np.linalg.norm(x_direction)
        y_direction = bounds[-1] - bounds[0]
        y_direction /= np.linalg.norm(y_direction)
        delta = 1000 * np.array(
            [
                -pad_x[0] * x_direction - pad_y[0] * y_direction,
                pad_x[1] * x_direction - pad_y[0] * y_direction,
                pad_x[1] * x_direction + pad_y[1] * y_direction,
                -pad_x[0] * x_direction + pad_y[1] * y_direction,
            ]
        )

        return self.__class__(bounds + delta)

    def shift(self, x_shift: float = 0, y_shift: float = 0) -> Self:
        """Translate the bounding box by specified distances.

        Parameters
        ----------
        x_shift : float, default 0
            Distance to shift in x direction in kilometers
        y_shift : float, default 0
            Distance to shift in y direction in kilometers

        Returns
        -------
        Self
            A new instance of the bounding box translated by the specified amounts
        """
        return self.__class__(self.bounds + 1000 * np.array([x_shift, y_shift]))

    def rotate(self, angle: float) -> Self:
        """Rotate the bounding box around its center by a specified angle.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees. Positive values indicate counterclockwise rotation.

        Returns
        -------
        Self
            A new instance of the bounding box rotated by the specified angle
        """
        rot_matrix = geo.rotation_matrix(np.radians(angle))
        origin = np.mean(self.bounds, axis=0)
        return self.__class__((self.bounds - origin) @ rot_matrix.T + origin)

    @classmethod
    def from_centroid_bearing_extents(
        cls,
        centroid: npt.ArrayLike,
        bearing: float,
        extent_x: float,
        extent_y: float,
    ) -> Self:
        """Create a bounding box from a centroid, bearing, and size.

        The x and y-directions are determined relative to the bearing,

                 N      y-direction = bearing
                 │    /
                 │   /
                 │  /
                 │ /
                 │/
                 ■
                  ╲
                   ╲
                    ╲
                     ╲ x-direction = bearing + 90

        Parameters
        ----------
        centroid : np.ndarray
            The centre of the bounding box (lat, lon).
        bearing : float
            A bearing from north for the bounding box, in degrees.
        extent_x : float
            The length along the x-direction of the bounding box, in
            kilometres.
        extent_y : float
            The length along the y-direction of the bounding box, in
            kilometres.

        Returns
        -------
        Self
            The bounding box with the given centre, bearing, and
            length along the x and y-directions.
        """
        centroid = np.asarray(centroid)
        corner_offset = (
            np.array(
                [[-1 / 2, -1 / 2], [1 / 2, -1 / 2], [1 / 2, 1 / 2], [-1 / 2, 1 / 2]]
            )
            * np.array([extent_y, extent_x])
            * 1000
        ) @ geo.rotation_matrix(np.radians(-bearing))
        return cls(coordinates.wgs_depth_to_nztm(centroid) + corner_offset)

    @classmethod
    def bounding_box_for_geometry(
        cls, geometry: shapely.Geometry, axis_aligned: bool = False
    ) -> Self:
        """Return a bounding box that minimally encloses a geometry.

        Parameters
        ----------
        geometry : shapely.Geometry
            The geometry to enclose.
        axis_aligned : bool
            If True, ensure that the bounding box is axis-aligned.

        Returns
        -------
        Self
            The bounding box for this geometry.

        Raises
        ------
        ValueError
            If the geometry does not have a well-defined bounding box.
            This occurs if the geometry is degenerate (either a line
            or a point).
        """
        if axis_aligned:
            bounding_box_polygon = shapely.envelope(geometry).normalize()
        else:
            bounding_box_polygon = shapely.oriented_envelope(geometry).normalize()
        if not (
            isinstance(bounding_box_polygon, shapely.Polygon)
            and len(bounding_box_polygon.exterior.coords) - 1 == 4
        ):
            raise ValueError("Ill-defined geometry for bounding box.")
        return cls(np.array(bounding_box_polygon.exterior.coords)[:-1])

    @classmethod
    def from_wgs84_coordinates(cls, corner_coordinates: npt.ArrayLike) -> Self:
        """Construct a bounding box from a list of corners.

        Parameters
        ----------
        corner_coordinates : np.ndarray
            The corners in (lat, lon) format.

        Returns
        -------
        Self
            The bounding box represented by these corners.
        """
        return cls(np.asarray(coordinates.wgs_depth_to_nztm(corner_coordinates)))

    @property
    def origin(self) -> npt.NDArray[np.float64]:  # numpydoc ignore=RT01
        """np.ndarray: The origin of the bounding box."""
        return coordinates.nztm_to_wgs_depth(np.mean(self.bounds, axis=0))

    @property
    def extent_x(self) -> np.float64:  # numpydoc ignore=RT01
        """float: The extent along the x-axis of the bounding box (in km)."""
        return np.linalg.norm(self.bounds[1] - self.bounds[0]) / 1000

    @property
    def extent_y(self) -> np.float64:  # numpydoc ignore=RT01
        """float: The extent along the y-axis of the bounding box (in km)."""
        return np.linalg.norm(self.bounds[2] - self.bounds[1]) / 1000

    @property
    def bearing(self) -> np.float64:  # numpydoc ignore=RT01
        """float: The bearing of the bounding box."""
        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        vertical_direction = np.append(self.bounds[-1] - self.bounds[0], 0)
        return geo.oriented_bearing_wrt_normal(
            north_direction, vertical_direction, up_direction
        )

    @property
    def great_circle_bearing(self) -> np.float64:  # numpydoc ignore=RT01
        """float: The great-circle bearing of the bounding box.

        This returns the bearing of the bounding box in WGS84
        coordinate space (as opposed to in the NZTM coordinate space).
        """
        return coordinates.nztm_bearing_to_great_circle_bearing(
            self.origin, self.extent_y / 2, self.bearing
        )

    @property
    def area(self) -> np.float64:  # numpydoc ignore=RT01
        """float: The area of the bounding box."""
        return self.extent_x * self.extent_y

    @property
    def polygon(self) -> Polygon:  # numpydoc ignore=RT01
        """Polygon: The shapely geometry for the bounding box."""
        return Polygon(np.append(self.bounds, np.atleast_2d(self.bounds[0]), axis=0))

    def contains(self, points: npt.ArrayLike) -> bool | npt.NDArray[np.bool_]:
        """Filter a list of points by whether they are contained in the bounding box.

        Parameters
        ----------
        points : np.array
            The points to filter.

        Returns
        -------
        bool or array of bools
            A boolean mask of the points in the bounding box.
        """
        points = np.asarray(points)
        offset = coordinates.wgs_depth_to_nztm(points) - self.bounds[0]
        frame = np.array(
            [self.bounds[1] - self.bounds[0], self.bounds[-1] - self.bounds[0]]
        )
        if offset.ndim > 1:
            offset = offset.T
        local_coordinates = np.linalg.solve(frame.T, offset)

        return np.all(
            ((local_coordinates > 0) | np.isclose(local_coordinates, 0, atol=1e-6))
            & ((local_coordinates < 1) | np.isclose(local_coordinates, 1, atol=1e-6)),
            axis=0,
        )

    def local_coordinates_to_wgs_depth(
        self,
        local_coordinates: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Convert bounding box coordinates to global coordinates.

        Parameters
        ----------
        local_coordinates : np.ndarray
            Local coordinates to convert. Local coordinates are 2D
            coordinates (x, y) given for a bounding box, where x
            represents displacement along the x-direction, and y
            displacement along the y-direction (see diagram below).

                                       1 1
                ┌─────────────────────┐ ^
                │                     │ │
                │                     │ │
                │                     │ │ +y
                │                     │ │
                │                     │ │
                └─────────────────────┘ │
             0 0   ─────────────────>
                         +x

        Returns
        -------
        np.ndarray
            An vector of (lat, lon) transformed coordinates.
        """
        frame = np.array(
            [self.bounds[1] - self.bounds[0], self.bounds[-1] - self.bounds[0]]
        )
        nztm_coords = self.bounds[0] + local_coordinates @ frame
        return coordinates.nztm_to_wgs_depth(nztm_coords)

    def wgs_depth_coordinates_to_local_coordinates(
        self, global_coordinates: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Convert coordinates (lat, lon) to bounding box coordinates (x, y).

        See `BoundingBox.local_coordinates_to_wgs_depth` for a description of
        bounding box coordinates.

        Parameters
        ----------
        global_coordinates : np.ndarray
            Global coordinates to convert.

        Returns
        -------
        np.ndarray
            Coordinates (x, y) representing the position of
            global_coordinates in bounding box coordinates.

        Raises
        ------
        ValueError
            If the given coordinates do not lie in the bounding box.
        """
        global_coordinates = np.asarray(global_coordinates)

        frame = np.array(
            [self.bounds[1] - self.bounds[0], self.bounds[-1] - self.bounds[0]]
        )
        offset = coordinates.wgs_depth_to_nztm(global_coordinates) - self.bounds[0]
        if offset.ndim > 1:
            offset = offset.T
        local_coordinates = np.linalg.solve(frame.T, offset)
        if not np.all(
            ((local_coordinates > 0) | np.isclose(local_coordinates, 0, atol=1e-6))
            & ((local_coordinates < 1) | np.isclose(local_coordinates, 1, atol=1e-6))
        ):
            raise ValueError("Specified coordinates do not lie in bounding box.")
        local_coordinates = np.clip(local_coordinates, 0, 1)
        return local_coordinates.T

    def __repr__(self):
        """A representation of the bounding box."""
        cls = self.__class__.__name__
        return f"{cls}(centre={self.origin}, bearing={self.bearing}, extent_x={self.extent_x}, extent_y={self.extent_y}, corners={self.corners})"


def minimum_area_bounding_box_for_polygons_masked(
    must_include: list[Polygon], may_include: list[Polygon], mask: Polygon
) -> BoundingBox:
    """Find a minimum area bounding box for the points must_include ∪ (may_include ∩ mask).

    Parameters
    ----------
    must_include : list[Polygon]
        List of polygons the bounding box must include.
    may_include : list[Polygon]
        List of polygons the bounding box will include portions of, when inside of mask.
    mask : Polygon
        The masking polygon.

    Returns
    -------
    BoundingBox
        The smallest box containing all the points of `must_include`, and all the
        points of `may_include` that lie within the bounds of `mask`.

    """
    may_include_polygon = shapely.normalize(shapely.union_all(may_include))
    must_include_polygon = shapely.normalize(shapely.union_all(must_include))
    bounding_polygon = shapely.normalize(
        shapely.union(
            must_include_polygon, shapely.intersection(may_include_polygon, mask)
        )
    )
    return BoundingBox.bounding_box_for_geometry(bounding_polygon)
