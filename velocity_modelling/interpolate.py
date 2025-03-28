"""
Interpolation functions for velocity modelling.

"""

import numba
import numpy as np


@numba.jit(nopython=True)
def linear_interpolation(p1: float, p2: float, v1: float, v2: float, p3: float):
    """
    Perform linear interpolation between two points.

    Parameters
    ----------
    p1 : float
        The first point.
    p2 : float
        The second point.
    v1 : float
        The value at the first point.
    v2 : float
        The value at the second point.
    p3 : float
        The point at which to interpolate.

    Returns
    -------
    float
        The interpolated value at point p3.
    """
    return v1 + (v2 - v1) * (p3 - p1) / (p2 - p1)


@numba.jit(nopython=True)
def linear_interpolation_vectorized(
    x0: float, x1: float, y0: np.ndarray | float, y1: np.ndarray | float, x: np.ndarray
) -> np.ndarray:
    """
    Perform linear interpolation using vectorized operations.

    Parameters
    ----------
    x0 : float
        First x coordinate.
    x1 : float
        Second x coordinate.
    y0 : np.ndarray or float
        First y coordinate(s).
    y1 : np.ndarray or float
        Second y coordinate(s).
    x : np.ndarray
        Array of x values to interpolate at.

    Returns
    -------
    np.ndarray
        Interpolated y values corresponding to x.
    """
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@numba.jit(nopython=True)
def bi_linear_interpolation(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    q11: float,
    q12: float,
    q21: float,
    q22: float,
    x: float,
    y: float,
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

    a = q11 * (x2 - x) * (y2 - y)
    b = q21 * (x - x1) * (y2 - y)
    c = q12 * (x2 - x) * (y - y1)
    d = q22 * (x - x1) * (y - y1)

    e = 1 / (dx * dy)

    return (a + b + c + d) * e
