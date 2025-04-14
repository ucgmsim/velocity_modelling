"""
Geotechnical Layer (GTL) velocity calculations module.

This module provides functions for velocity and density calculations based on
Brocher correlations and the Ely (2010) GTL model. It includes both scalar and
vectorized implementations for efficiency.

All velocity values in this module use km/s as units unless otherwise specified.
Depths are in metres, and densities are in g/cm^3.
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def rho_from_vp_brocher(vp: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate density given P-wave velocity using the Brocher correlation.

     Parameters
     ----------
     vp : float or np.ndarray
         Primary wave velocity (km/s).

     Returns
     -------
     float or np.ndarray
         Density (g/cm^3) from the Brocher correlation.
    """
    density = vp * (
        1.6612 + vp * (-0.4721 + vp * (0.0671 + vp * (-0.0043 + 0.000106 * vp)))
    )
    return density


@numba.jit(nopython=True)
def vp_from_vs_brocher(vs: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate P-wave velocity given S-wave velocity using the Brocher correlation.

     Parameters
     ----------
     vs : float or np.ndarray
         Secondary wave velocity (km/s).

     Returns
     -------
     float or np.ndarray
         Primary wave velocity (km/s) from the Brocher correlation.
    """
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4
    return vp


@numba.jit(nopython=True)
def vs_from_vp_brocher(vp: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate S-wave velocity given P-wave velocity using the Brocher correlation.

    Parameters
    ----------
    vp : float or np.ndarray
        Primary wave velocity (km/s).

    Returns
    -------
    float or np.ndarray
        Secondary wave velocity (km/s) from the Brocher correlation.
    """
    vs = 0.7858 - 1.2344 * vp + 0.7949 * vp**2 - 0.1238 * vp**3 + 0.0064 * vp**4
    return vs


@numba.jit(nopython=True)
def v30gtl_vectorized(
    vs30: float | np.ndarray, vt: np.ndarray, z: np.ndarray, zt: float
):
    """
    Vectorized VS30 Geotechnical Layer (GTL) velocity adjustment based on Ely (2010).

    This function is optimized for processing multiple depth points simultaneously.

    Parameters
    ----------
    vs30 : float or np.ndarray
        VS30 value (m/s).
    vt : np.ndarray
        Array of target velocity values (km/s).
    z : np.ndarray
        Array of depth values (m), must be positive.
    zt : float
        Taper depth (m).

    Returns
    -------
    tuple
        (vs, vp, rho): Arrays of adjusted S-wave velocities (km/s),
        P-wave velocities (km/s), and densities (g/cm^3).
    """
    # Constants
    a = 0.5
    b = 2.0 / 3.0
    c = 2.0

    # Normalize depth
    z_normalized = z / zt  # Shape: (n,)

    # Vectorized computation of f and g
    f = z_normalized + b * (z_normalized - z_normalized * z_normalized)
    g = (
        a
        - (a + 3.0 * c) * z_normalized
        + c * z_normalized * z_normalized
        + 2.0 * c * np.sqrt(z_normalized)
    )

    # Adjust vs
    vs = f * vt + g * (vs30 / 1000.0)  # vs30 converted to km/s

    # Compute vp from vs using Brocher correlation (vectorized)
    vp = vp_from_vs_brocher(vs)

    # Compute rho from vp using Brocher correlation (vectorized)
    rho = rho_from_vp_brocher(vp)

    return vs, vp, rho
