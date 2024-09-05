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

import dataclasses
from typing import Self

import numpy as np
import numpy.typing as npt
import scipy as sp
import shapely
from shapely import Polygon

from qcore import coordinates, geo


@dataclasses.dataclass
class BoundingBox:
    """Represents a 2D bounding box with properties and methods for calculations.

    Attributes
    ----------
    corners : np.ndarray
       The corners of the bounding box in cartesian coordinates. The
       order of the corners should be counter clock-wise from the bottom-left point
       (minimum x, minimum y).
    """

    bounds: npt.NDArray[np.float64]

    @property
    def corners(self) -> np.ndarray:
        """np.ndarray: the corners of the bounding box in (lat, lon) format."""
        return coordinates.nztm_to_wgs_depth(self.bounds)

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
            * np.array([extent_x, extent_y])
            * 1000
        ) @ geo.rotation_matrix(-np.radians(bearing))
        return cls(coordinates.wgs_depth_to_nztm(centroid) + corner_offset)

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
    def origin(self) -> npt.NDArray[np.float64]:
        """np.ndarray: The origin of the bounding box."""
        return coordinates.nztm_to_wgs_depth(np.mean(self.bounds, axis=0))

    @property
    def extent_x(self) -> np.float64:
        """float: The extent along the x-axis of the bounding box (in km)."""
        return np.linalg.norm(self.bounds[1] - self.bounds[0]) / 1000

    @property
    def extent_y(self) -> np.float64:
        """float: The extent along the y-axis of the bounding box (in km)."""
        return np.linalg.norm(self.bounds[2] - self.bounds[1]) / 1000

    @property
    def bearing(self) -> np.float64:
        """float: The bearing of the bounding box."""
        north_direction = np.array([1, 0, 0])
        up_direction = np.array([0, 0, 1])
        horizontal_direction = np.append(self.bounds[1] - self.bounds[0], 0)
        return geo.oriented_bearing_wrt_normal(
            north_direction, horizontal_direction, up_direction
        )

    @property
    def area(self) -> np.float64:
        """float: The area of the bounding box."""
        return self.extent_x * self.extent_y

    @property
    def polygon(self) -> Polygon:
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
            [self.bounds[1] - self.bounds[0], self.bounds[2] - self.bounds[0]]
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
        local_coords: npt.ArrayLike,
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
            [self.bounds[1] - self.bounds[0], self.bounds[2] - self.bounds[0]]
        )
        nztm_coords = self.bounds[0] + local_coords @ frame
        return coordinates.nztm_to_wgs_depth(nztm_coords)

    def wgs_depth_coordinates_to_local_coordinates(
        self, global_coords: npt.ArrayLike
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
        global_coords = np.asarray(global_coords)

        frame = np.array(
            [self.bounds[1] - self.bounds[0], self.bounds[2] - self.bounds[0]]
        )
        offset = coordinates.wgs_depth_to_nztm(global_coords) - self.bounds[0]
        if offset.ndim > 1:
            offset = offset.T
        local_coordinates = np.linalg.solve(frame.T, offset)
        if not np.all(
            ((local_coordinates > 0) | np.isclose(local_coordinates, 0, atol=1e-6))
            & ((local_coordinates < 1) | np.isclose(local_coordinates, 1, atol=1e-6))
        ):
            raise ValueError("Specified coordinates do not lie in bounding box.")

        return np.clip(local_coordinates, 0, 1)


def axis_aligned_bounding_box(points: npt.NDArray[np.float64]) -> BoundingBox:
    """Find the axis-aligned bounding box containing points.

    Parameters
    ----------
    points : np.ndarray
        The points to bound.

    Returns
    -------
    BoundingBox
        The axis-aligned bounding box.
    """
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    corners = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
    return BoundingBox(corners)


def minimum_area_bounding_box(points: npt.NDArray[np.float64]) -> BoundingBox:
    """Find the smallest rectangle bounding points. The rectangle may be rotated.

    Parameters
    ----------
    points : np.ndarray
        The points to bound.

    Returns
    -------
    BoundingBox
        The minimum area bounding box.
    """
    # This is a somewhat brute-force method to obtain the minimum-area bounding
    # box of a set of points, where the bounding box is not axis-aligned and is
    # instead allowed to be rotated. The idea is to reduce the problem to the
    # far simpler axis-aligned bounding box by observing that the minimum
    # area bounding box must have a side parallel with *some* edge of the
    # convex hull of the points. By rotating the picture so that the shared
    # edge is axis-aligned, the problem is reduced to that of finding the
    # axis-aligned bounding box. Because we do not know this edge apriori,
    # we simply try it for all the edges and then take the smallest area
    # box at the end.
    convex_hull = sp.spatial.ConvexHull(points).points
    segments = np.array(
        [
            convex_hull[(i + 1) % len(convex_hull)] - convex_hull[i]
            for i in range(len(convex_hull))
        ]
    )
    # This finds the slope of each segment with respect to the axes.
    rotation_angles = -np.arctan2(segments[:, 1], segments[:, 0])

    # Create a list of rotated bounding boxes by rotating each rotation angle,
    # and then finding the axis-aligned bounding box of the convex hull. This
    # creates a list of boxes that are each parallel to a different segment.
    bounding_boxes = [
        axis_aligned_bounding_box(convex_hull @ geo.rotation_matrix(angle).T)
        for angle in rotation_angles
    ]

    minimum_rotation_angle, minimum_bounding_box = min(
        zip(rotation_angles, bounding_boxes), key=lambda rot_box: rot_box[1].area
    )
    # axis-aligned bounding is not always included in the above
    # search, so we should check against this too!
    aa_box = axis_aligned_bounding_box(convex_hull)
    if aa_box.area < minimum_bounding_box.area:
        return aa_box

    # rotating by -minimum_rotation_angle we undo the rotation applied
    # to obtain bounding_boxes.
    rotation_matrix = geo.rotation_matrix(-minimum_rotation_angle).T
    corners = minimum_bounding_box.bounds @ rotation_matrix
    return BoundingBox(corners)


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

    if isinstance(bounding_polygon, Polygon):
        return minimum_area_bounding_box(np.array(bounding_polygon.exterior.coords))
    return minimum_area_bounding_box(
        np.vstack([np.array(geom.exterior.coords) for geom in bounding_polygon.geoms])
    )
