from numba import njit
import numpy as np

from velocity_modelling.cvm.constants import EARTH_RADIUS_MEAN


@njit
def lat_lon_to_distance(
    lat_lon_array: np.ndarray, origin_lat: float, origin_lon: float
) -> np.ndarray:
    """
    Calculate the distance from an origin point to a set of latitude and longitude points.

    Parameters
    ----------
    lat_lon_array : np.ndarray
        An array of shape (N, 2) where N is the number of points. Each row contains [latitude, longitude].
    origin_lat : float
        The latitude of the origin point.
    origin_lon : float
        The longitude of the origin point.

    Returns
    -------
    np.ndarray
        An array of distances from the origin point to each point in lat_lon_array.
    """
    ref_lon = np.deg2rad(origin_lon)
    ref_lat = np.deg2rad(origin_lat)

    lat_rad = np.deg2rad(lat_lon_array[:, 0])
    lon_rad = np.deg2rad(lat_lon_array[:, 1])

    dLon = lon_rad - ref_lon

    dz = np.sin(lat_rad) - np.sin(ref_lat)
    dx = np.cos(dLon) * np.cos(lat_rad) - np.cos(ref_lat)
    dy = np.sin(dLon) * np.cos(lat_rad)

    distances = (
        np.arcsin(np.sqrt(dx * dx + dy * dy + dz * dz) / 2) * 2 * EARTH_RADIUS_MEAN
    )

    return distances
