import numpy as np

from velocity_modelling.geometry import (
    compute_rotation_matrix,
    geographic_to_rotated_spherical,
)


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def geographic_to_rotated(lat, lon, origin_lat, origin_lon, origin_rot):
    KM_PER_DEGREE = (2 * np.pi * 6378.139) / 360
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    origin_lat_rad = np.radians(origin_lat)
    origin_lon_rad = np.radians(origin_lon)
    rot_rad = np.radians(-origin_rot)

    x = (lon - origin_lon) * KM_PER_DEGREE * np.cos(origin_lat_rad)
    y = (lat - origin_lat) * KM_PER_DEGREE

    cos_rot = np.cos(rot_rad)
    sin_rot = np.sin(rot_rad)
    x_rot = x * cos_rot + y * sin_rot
    y_rot = -x * sin_rot + y * cos_rot

    rot_lat = origin_lat + y_rot / KM_PER_DEGREE
    rot_lon = origin_lon + x_rot / (KM_PER_DEGREE * np.cos(origin_lat_rad))

    return rot_lat, rot_lon


def test_identity_at_north_pole():
    """Test that the transformation is an identity map when origin is at North Pole with no rotation."""
    lat2, lon2 = geographic_to_rotated_spherical(10.0, 20.0, 90.0, 0.0, 0.0)
    assert approx(lat2, 10.0)
    assert approx(lon2, 20.0)

    lat2, lon2 = geographic_to_rotated_spherical(10.0, 20.0, 90.0, 180.0, 0.0)
    assert approx(lat2, 10.0)
    assert approx(lon2, 200.0, tol=1e-5)  # Relaxed tolerance due to numerical precision


def test_longitude_spin_about_pole():
    """Test that rotation around the North Pole shifts longitude correctly."""
    # origin_rot = 90.0 means counterclockwise rotation by 90°
    lat2, lon2 = geographic_to_rotated_spherical(0.0, 1.0, 90.0, 0.0, 90.0)
    assert approx(lat2, 0.0)
    assert approx((lon2 + 360) % 360, 271.0)  # 1 - 90 = -89, normalized to 271


def test_origin_becomes_pole():
    """Test that the rotation origin maps to the new pole."""
    lat2, lon2 = geographic_to_rotated_spherical(30.0, 40.0, 30.0, 40.0, 0.0)
    assert approx(lat2, 90.0)
    assert 0.0 <= lon2 < 360.0


def test_rotation_direction():
    """Test the direction of rotation to confirm counterclockwise convention."""
    # origin_rot = 90.0 means counterclockwise -90°
    lat2, lon2 = geographic_to_rotated_spherical(0.0, 1.0, 90.0, 0.0, 90.0)
    assert approx(lat2, 0.0)
    assert approx((lon2 + 360) % 360, 271.0)  # 1 - 90 = -89, normalized to 271

    # origin_rot = -90.0 means clockwise +90°
    lat2, lon2 = geographic_to_rotated_spherical(0.0, 1.0, 90.0, 0.0, -90.0)
    assert approx(lat2, 0.0)
    assert approx((lon2 + 360) % 360, 91.0)  # 1 + 90 = 91


def test_velocity_model_coordinates():
    """Test with actual velocity model parameters and cross-section coordinates."""
    origin_lat = -43.35805601377026
    origin_lon = 171.7875594038442
    origin_rot = 38.3183225905264
    lat, lon = -44.3, 170.3

    expected_lat, expected_lon = geographic_to_rotated(
        lat, lon, origin_lat, origin_lon, origin_rot
    )
    lat2, lon2 = geographic_to_rotated_spherical(
        lat, lon, origin_lat, origin_lon, origin_rot
    )
    assert approx(lat2, expected_lat, tol=1e-3)
    assert approx(lon2, expected_lon, tol=1e-3)


def test_near_pole_numerical_stability():
    """Test numerical stability near the pole."""
    lat2, lon2 = geographic_to_rotated_spherical(89.999, 0.0, 90.0, 0.0, 0.0)
    assert approx(lat2, 89.999, tol=1e-3)
    assert 0.0 <= lon2 < 360.0


def test_compute_rotation_matrix_identity():
    """Test that compute_rotation_matrix produces an identity-like transformation at North Pole with no rotation."""
    amat = compute_rotation_matrix(90.0, 0.0, 0.0)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(amat, expected, atol=1e-6)
