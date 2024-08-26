from typing import Optional

import numpy as np
from hypothesis import HealthCheck, assume, given, seed, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nst

from qcore import coordinates
from velocity_modelling import bounding_box
from velocity_modelling.bounding_box import BoundingBox


def coordinate(lat: float, lon: float, depth: Optional[float] = None) -> np.ndarray:
    if depth is not None:
        return np.array([lat, lon, depth])
    return np.array([lat, lon])


def coordinate_hashable(
    lat: float, lon: float, depth: Optional[float] = None
) -> tuple[float, ...]:
    if depth is not None:
        return (np.round(lat, 5), np.round(lon, 5), np.round(depth, 5))
    return (np.round(lat, 5), np.round(lon, 5))


def valid_coordinates(point_coordinates: np.ndarray) -> bool:
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
    box = bounding_box.minimum_area_bounding_box(points)
    assert box.contains(points)


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
        extent_x=st.floats(1, 30, allow_nan=False, allow_infinity=False),
        extent_y=st.floats(1, 30, allow_nan=False, allow_infinity=False),
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
    points = np.array(points)
    assume(
        np.linalg.matrix_rank(
            coordinates.wgs_depth_to_nztm(np.c_[points, np.ones_like(points[:, 1])])
        )
        == 3
    )

    box = bounding_box.minimum_area_bounding_box(points)
    aa_box = bounding_box.axis_aligned_bounding_box(
        coordinates.wgs_depth_to_nztm(points)
    )

    assume(dummy_bounding_box.contains(points))
    assert box.area < aa_box.area or np.isclose(aa_box.area, box.area)
    assert box.area < dummy_bounding_box.area or np.isclose(
        box.area, dummy_bounding_box.area
    )
