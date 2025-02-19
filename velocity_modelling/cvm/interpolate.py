import numpy as np
import numba


@numba.jit(nopython=True)
def linear_interpolation(
    p1: np.float64, p2: np.float64, v1: np.float64, v2: np.float64, p3: np.float64
):
    """
    Perform linear interpolation between two points.

    Parameters
    ----------
    p1 : np.float64
        The first point.
    p2 : np.float64
        The second point.
    v1 : np.float64
        The value at the first point.
    v2 : np.float64
        The value at the second point.
    p3 : np.float64
        The point at which to interpolate.

    Returns
    -------
    np.float64
        The interpolated value at point p3.
    """
    return v1 + (v2 - v1) * (p3 - p1) / (p2 - p1)


# @numba.jit(nopython=True)
def bi_linear_interpolation(
    x1: np.float64,
    x2: np.float64,
    y1: np.float64,
    y2: np.float64,
    q11: np.float64,
    q12: np.float64,
    q21: np.float64,
    q22: np.float64,
    x: np.float64,
    y: np.float64,
):
    """
    Perform bilinear interpolation between four points.

    Parameters
    ----------
    x1, x2, y1, y2 : float
        Coordinates of the points.
    q11, q12, q21, q22 : float
        Values at the four points.
    x, y : float
        Coordinates of the point to interpolate.

    Returns
    -------
    float
        Interpolated value at the given x, y.
    """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 or dy == 0:
        raise ValueError("Error: Zero division in bi-linear interpolation.")

    A = q11 * (x2 - x) * (y2 - y)
    B = q21 * (x - x1) * (y2 - y)
    C = q12 * (x2 - x) * (y - y1)
    D = q22 * (x - x1) * (y - y1)

    E = 1 / (dx * dy)

    return (A + B + C + D) * E
