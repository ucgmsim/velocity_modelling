from typing import Optional

import numpy as np
import pytest
import shapely
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from qcore import coordinates
from velocity_modelling import bounding_box
from velocity_modelling.bounding_box import BoundingBox


def coordinate(lat: float, lon: float, depth: Optional[float] = None) -> np.ndarray:
    """Create a coordinate array.

    Parameters
    ----------
    lat : float
        The latitude of the coordinate
    lon : float
        The longitude of the coordinate.
    depth : Optional[float]
        The optional depth of the coordinate.

    Returns
    -------
    np.ndarray
        An array of [lat, lon, depth].
    """
    if depth is not None:
        return np.array([lat, lon, depth])
    return np.array([lat, lon])


def coordinate_hashable(
    lat: float, lon: float, depth: Optional[float] = None
) -> tuple[float, ...]:
    """Create a hashable coordinate tuple.

    Parameters
    ----------
    lat : float
        The latitude of the coordinate.
    lon : float
        The longitude of the coordinate.
    depth : Optional[float]
        The optional depth of the coordinate.

    Returns
    -------
    tuple[float, ...]
        A tuple (lat, lon, depth). The coordinates are rounded to 5dp
        to encourage random sampling to generate points farther apart.

    """
    if depth is not None:
        return (np.round(lat, 5), np.round(lon, 5), np.round(depth, 5))
    return (np.round(lat, 5), np.round(lon, 5))


def valid_coordinates(point_coordinates: np.ndarray) -> bool:
    """Test if coordinates are valid (i.e. in the bounds for NZTM).

    Parameters
    ----------
    point_coordinates : np.ndarray
        The coordinates to check.

    Returns
    -------
    bool
        True if the coordinates are in bounds for NZTM.
    """
    try:
        return np.all(np.isfinite(coordinates.wgs_depth_to_nztm(point_coordinates)))
    except ValueError:
        return False


@given(
    centroid=st.builds(
        coordinate,
        lat=st.floats(-50, -31),
        lon=st.floats(160, 180),
    ),
    bearing=st.floats(0, 90, exclude_max=True),
    extent_x=st.floats(0.1, 1000, allow_nan=False, allow_infinity=False),
    extent_y=st.floats(0.1, 1000, allow_nan=False, allow_infinity=False),
)
def test_bounding_box_construction(
    centroid: np.ndarray, bearing: float, extent_x: float, extent_y: float
) -> None:
    """Test that the bounding box centroid constructor produces a box with required values."""
    box = BoundingBox.from_centroid_bearing_extents(
        centroid, bearing, extent_x, extent_y
    )
    assume(valid_coordinates(box.corners))
    assert np.allclose(box.origin, centroid)
    # A box's bearing is not uniquely defined!
    assert np.isclose(box.bearing % 90, bearing % 90, atol=1e-1) or (
        np.isclose(box.bearing, 0) and np.isclose(bearing % 90, 90)
    )
    assert np.allclose(
        [box.extent_x, box.extent_y], [extent_x, extent_y]
    ) or np.allclose([box.extent_x, box.extent_y], [extent_y, extent_x])
    assert np.isclose(box.area, box.extent_x * box.extent_y)

    assert np.allclose(
        np.sort(np.array(box.polygon.exterior.coords)[:-1], axis=0),
        np.sort(box.bounds, axis=0),
    )
    # Check that the sorted order of the centroid constructed box
    # matches the corner coordinate constructed box, and that the
    # corner coordinate construction works.
    assert np.allclose(
        np.sort(BoundingBox.from_wgs84_coordinates(box.corners).corners, axis=-1),
        np.sort(box.corners, axis=-1),
    )
    # This just checks that the repr doesn't cause a crash. The value
    # is insignificant as it is just used for debugging.
    _ = repr(box)


@given(
    box=st.builds(
        BoundingBox.from_centroid_bearing_extents,
        centroid=st.builds(
            coordinate,
            lat=st.floats(-50, -31),
            lon=st.floats(160, 180),
        ),
        bearing=st.floats(0, 360),
        extent_x=st.floats(0.1, 1000, allow_nan=False, allow_infinity=False),
        extent_y=st.floats(0.1, 1000, allow_nan=False, allow_infinity=False),
    ),
    local_x=st.floats(0, 1),
    local_y=st.floats(0, 1),
)
@settings(deadline=None)
def test_bounding_box_containment(box: BoundingBox, local_x: float, local_y: float):
    assert box.contains(
        coordinates.nztm_to_wgs_depth(
            box.bounds[0]
            + local_x * (box.bounds[1] - box.bounds[0])
            + local_y * (box.bounds[2] - box.bounds[1])
        )
    )


@given(
    points=st.lists(
        elements=st.builds(
            coordinate_hashable,
            lat=st.floats(-50, -31),
            lon=st.floats(160, 180),
        ),
        min_size=4,
        unique=True,
    )
)
def test_minimum_bounding_box_containment(points: list[np.ndarray]):
    points = coordinates.wgs_depth_to_nztm(np.array(points))
    # The QuickHull algorithm used to construct the minimum area
    # bounding box assumes that the points are not all collinear.
    assume(np.linalg.matrix_rank(np.c_[points, np.ones_like(points[:, 1])]) == 3)
    box = BoundingBox.bounding_box_for_geometry(shapely.MultiPoint(points))

    assert box.contains(coordinates.nztm_to_wgs_depth(points)).all()


@given(
    points=st.lists(
        elements=st.builds(
            coordinate_hashable,
            lat=st.floats(-50, -31),
            lon=st.floats(160, 180),
        ),
        min_size=4,
        unique=True,
    )
)
def test_minimum_bounding_box_minimality(points: list[np.ndarray]):
    """Tests the minimality by assuming that the shapely implementation is correct,
    and then asserting that the geometry is close to the minimum area geometry."""
    points = coordinates.wgs_depth_to_nztm(np.array(points))
    # The QuickHull algorithm used to construct the minimum area
    # bounding box assumes that the points are not all collinear.
    assume(np.linalg.cond(np.c_[points, np.ones_like(points[:, 1])]) < 1e8)
    box = BoundingBox.bounding_box_for_geometry(shapely.MultiPoint(points))
    bounding_box_polygon = shapely.Polygon(box.bounds).normalize()
    minimum_envelope = shapely.oriented_envelope(shapely.MultiPoint(points)).normalize()
    assert bounding_box_polygon.area == pytest.approx(minimum_envelope.area)


def test_masked_bounding_box():
    r"""Check that the masked bounding box works in a simple test.

    We construct a scenario with three overlapping circles

     must include    mask     may include
         -------   -------     -------
       -/...... --/       \---/       \--
      /....... /..\        /*\           \
     /......../....\      /***\           \
     |........|....|      |***|           |
     \........\..../      \***/           /
      \........\../        \*/           /
       -\.......--\       /---\       /--
         -------   -------     -------
           -1500      0          +1500
        <------------ x ------------->
    The bounding box should include everything inside the first
    circle, but only the intersection of mask and the third circle.

    """
    centre = coordinates.wgs_depth_to_nztm(np.array([-43, 172]))
    must_include_points = shapely.buffer(
        shapely.Point(centre), 1000
    )  # 1km circle from centre
    mask = shapely.buffer(shapely.Point(centre + np.array([1500, 0])), 1000)
    may_include_points = shapely.buffer(
        shapely.Point(centre + np.array([3000, 0])), 1000
    )
    box = bounding_box.minimum_area_bounding_box_for_polygons_masked(
        [must_include_points], [may_include_points], mask
    )
    theta = np.linspace(0, 2 * np.pi)
    r = np.linspace(0, 1)
    r_grid, t_grid = np.meshgrid(theta, r)
    circle_coordinates = np.vstack([r_grid.ravel(), t_grid.ravel()])

    circle_points = circle_coordinates[1] * np.vstack(
        (np.cos(circle_coordinates[0]), np.sin(circle_coordinates[0]))
    )

    # There are some issues with the circular approximation used by
    # shapely, so it will not contain all points right at the edge.
    test_must_include_points = centre + 995 * circle_points.T
    test_may_include_points = centre + np.array([3000, 0]) + 1000 * circle_points.T

    test_may_include_points_mask = np.array(
        [mask.contains(shapely.Point(p)) for p in test_may_include_points]
    )
    test_may_include_has_points = test_may_include_points[test_may_include_points_mask]
    test_not_include_points = test_may_include_points[~test_may_include_points_mask]

    # box contains all the points in the must include circle
    assert box.contains(coordinates.nztm_to_wgs_depth(test_must_include_points)).all()
    # box contains all points in may include also in the mask
    assert box.contains(
        coordinates.nztm_to_wgs_depth(test_may_include_has_points)
    ).all()
    # Can't assert that it does not contain *any* of the points in the
    # rest of the may include circle, but it is definitely wrong to
    # contain all of them
    assert not box.contains(
        coordinates.nztm_to_wgs_depth(test_not_include_points)
    ).all()


@given(
    box=st.builds(
        BoundingBox.from_centroid_bearing_extents,
        centroid=st.builds(
            coordinate,
            lat=st.floats(-47, -40),
            lon=st.floats(166, 177),
        ),
        bearing=st.floats(0, 360),
        extent_x=st.floats(20, 1000, allow_nan=False, allow_infinity=False),
        extent_y=st.floats(20, 1000, allow_nan=False, allow_infinity=False),
    ),
    local_x=st.floats(0, 1),
    local_y=st.floats(0, 1),
)
def test_bounding_box_grid_coordinates_inversion(
    box: BoundingBox, local_x: float, local_y: float
):
    local_coords = np.array([local_x, local_y])
    assert np.allclose(
        box.wgs_depth_coordinates_to_local_coordinates(
            box.local_coordinates_to_wgs_depth(local_coords)
        ),
        local_coords,
        atol=0.01,
    )


@given(
    box=st.builds(
        BoundingBox.from_centroid_bearing_extents,
        centroid=st.builds(
            coordinate,
            lat=st.floats(-47, -40),
            lon=st.floats(166, 177),
        ),
        bearing=st.floats(0, 360),
        extent_x=st.floats(20, 1000, allow_nan=False, allow_infinity=False),
        extent_y=st.floats(20, 1000, allow_nan=False, allow_infinity=False),
    )
)
def test_bounding_box_boundary_values(box: BoundingBox):
    assert np.allclose(
        box.wgs_depth_coordinates_to_local_coordinates(box.corners),
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    )


@given(
    box=st.builds(
        BoundingBox.from_centroid_bearing_extents,
        centroid=st.builds(
            coordinate,
            lat=st.floats(-47, -40),
            lon=st.floats(166, 177),
        ),
        bearing=st.floats(0, 360),
        extent_x=st.floats(20, 1000, allow_nan=False, allow_infinity=False),
        extent_y=st.floats(20, 1000, allow_nan=False, allow_infinity=False),
    ),
    local_x=st.floats(0, 1),
    local_y=st.floats(0, 1),
)
def test_bounding_box_containment_consistency(
    box: BoundingBox, local_x: float, local_y: float
):
    local_coordinates = np.array([local_x, local_y])
    assert box.contains(box.local_coordinates_to_wgs_depth(local_coordinates))


@pytest.mark.parametrize(
    "box,expected_bearing",
    [
        (
            BoundingBox.from_centroid_bearing_extents(
                np.array([-45.74097357516373, 167.209054441501]), 0, 100, 150
            ),
            4.117063885773218,
        ),
        (
            BoundingBox.from_centroid_bearing_extents(
                np.array([-45.74097357516373, 167.209054441501]), 30, 100, 150
            ),
            34.04656913667332,
        ),
        (
            BoundingBox.from_centroid_bearing_extents(
                np.array([-37.06313846120225, 174.93752057998879]), 23, 100, 80
            ),
            21.751793740814296,
        ),
    ],
)
def test_bounding_box_great_circle_bearing(box: BoundingBox, expected_bearing: float):
    assert box.great_circle_bearing == pytest.approx(expected_bearing)
