from typing import Optional

import numpy as np
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
    bearing=st.floats(0, 360),
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
    assert np.isclose(box.bearing, bearing, atol=1e-1) or (
        np.isclose(box.bearing, 0) and np.isclose(bearing, 360)
    )
    assert np.isclose(box.extent_x, extent_x)
    assert np.isclose(box.extent_y, extent_y)
    assert np.isclose(box.area, box.extent_x * box.extent_y)


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
def test_bounding_box_containment(box: BoundingBox, local_x, local_y):
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
    points = np.array(points)
    # The QuickHull algorithm used to construct the minimum area
    # bounding box assumes that the points are not all collinear.
    assume(
        np.linalg.matrix_rank(
            coordinates.wgs_depth_to_nztm(np.c_[points, np.ones_like(points[:, 1])])
        )
        == 3
    )
    box = bounding_box.minimum_area_bounding_box(coordinates.wgs_depth_to_nztm(points))
    assert box.contains(points).all()


# Slower test!
@given(
    points=st.lists(
        elements=st.builds(
            coordinate_hashable, lon=st.floats(173, 173.1), lat=st.floats(-43.1, -43)
        ),
        min_size=4,
        unique=True,
    ),
    dummy_bounding_box=st.builds(
        BoundingBox.from_centroid_bearing_extents,
        centroid=st.builds(
            coordinate,
            lon=st.floats(173, 173.1),
            lat=st.floats(-43.1, -43),
        ),
        bearing=st.floats(0, 360),
        # In my testing, you need to allow really big boxes to get
        # enough of a sample to test minimality.
        extent_x=st.floats(9, 30, allow_nan=False, allow_infinity=False),
        extent_y=st.floats(9, 30, allow_nan=False, allow_infinity=False),
    ),
)
# Allow a lot of generated examples to get cases where a random box is
# pretty good to properly test the minimum area bounding box.
@settings(max_examples=500, suppress_health_check=[HealthCheck.filter_too_much])
# seed = 1 so that GA runs are deterministic.
@seed(1)
def test_minimum_bounding_box_minimality(
    points: list[np.ndarray], dummy_bounding_box: BoundingBox
):
    """Check that the minimum box is minimal in area.

    This test heuristically evaluates the minimality of the bounding box by:

    1. Generating a random set of points in a small area.
    2. Generating a random bounding box
    3. Finding the minimum area bounding box via minimum_area_bounding_box
    4. Confirming that the area of this box is smaller than:
       - The axis-aligned minimum area bounding box of these points, and
       - The randomly sampled bounding box.
    """
    points = np.array(points)
    assume(
        np.linalg.matrix_rank(
            coordinates.wgs_depth_to_nztm(np.c_[points, np.ones_like(points[:, 1])])
        )
        == 3
    )

    box = bounding_box.minimum_area_bounding_box(coordinates.wgs_depth_to_nztm(points))
    aa_box = bounding_box.axis_aligned_bounding_box(
        coordinates.wgs_depth_to_nztm(points)
    )

    assume(dummy_bounding_box.contains(points).all())
    assert box.area < aa_box.area or np.isclose(aa_box.area, box.area)
    assert box.area < dummy_bounding_box.area or np.isclose(
        box.area, dummy_bounding_box.area
    )


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
    R, T = np.meshgrid(theta, r)
    circle_coordinates = np.vstack([R.ravel(), T.ravel()])

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
