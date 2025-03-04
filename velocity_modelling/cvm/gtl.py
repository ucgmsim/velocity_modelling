import numpy as np
import numba


@numba.jit(nopython=True)
def rho_from_vp_brocher(vp: np.float64) -> np.float64:
    """
    Calculate Rho given Vp from the Brocher correlation.

    Parameters
    ----------
    vp : float
        Primary wave velocity.

    Returns
    -------
    float
        Density (Rho) from the Brocher correlation.
    """
    density = vp * (
        1.6612 + vp * (-0.4721 + vp * (0.0671 + vp * (-0.0043 + 0.000106 * vp)))
    )
    return density


@numba.jit(nopython=True)
def vp_from_vs_brocher(vs: np.float64) -> np.float64:
    """
    Calculate Vp given vs from the Brocher correlation.

    Parameters
    ----------
    vs : float
        Secondary wave velocity.

    Returns
    -------
    float
        Primary wave velocity (Vp) from the Brocher correlation.
    """
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4
    return vp


@numba.jit(nopython=True)
def vs_from_vp_brocher(vp: np.float64) -> np.float64:
    """
    Calculate vs given Vp from the Brocher correlation.

    Parameters
    ----------
    vp : float
        Primary wave velocity.

    Returns
    -------
    float
        Secondary wave velocity (vs) from the Brocher correlation.
    """
    vs = 0.7858 - 1.2344 * vp + 0.7949 * vp**2 - 0.1238 * vp**3 + 0.0064 * vp**4
    return vs


@numba.jit(nopython=True)
def v30gtl(vs30: np.float64, vt: np.float64, z: np.float64, zt: np.float64):
    """
    vs30 Geotechnical Layer (GTL) based on Ely (2010).

    Parameters
    ----------
    vs30 : float
        vs30 value.
    vt : float
        Velocity value.
    z : float
        Depth value.
    zt : float
        Taper depth.

    Returns
    -------
    float
        vs value.
    float
        vp value.
    float
        rho value.


    """
    a = 0.5
    b = 2.0 / 3.0
    c = 2

    z = z / zt  # z must be positive here
    f = z + b * (z - z * z)
    g = a - (a + 3.0 * c) * z + c * z * z + 2.0 * c * np.sqrt(z)

    vs = f * vt + g * vs30 / 1000  # vs30 must be in km/s
    vp = vp_from_vs_brocher(vs)
    rho = rho_from_vp_brocher(vp)

    return vs, vp, rho


@numba.jit(nopython=True)
def v30gtl_vectorized(vs30: np.float64, vt: np.ndarray, z: np.ndarray, zt: np.float64):
    """
    Vectorized vs30 Geotechnical Layer (GTL) based on Ely (2010).

    Parameters
    ----------
    vs30 : float
        vs30 value (in m/s).
    vt : np.ndarray
        Array of velocity values (in km/s).
    z : np.ndarray
        Array of depth values (positive, in meters).
    zt : float
        Taper depth (in meters).

    Returns
    -------
    tuple
        (vs, vp, rho) arrays of adjusted vs, vp, and rho values.
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
    vp = 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4

    # Compute rho from vp using Brocher correlation (vectorized)
    rho = vp * (
        1.6612 + vp * (-0.4721 + vp * (0.0671 + vp * (-0.0043 + 0.000106 * vp)))
    )

    return vs, vp, rho
