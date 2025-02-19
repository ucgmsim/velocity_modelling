import numpy as np


def rho_from_vp_brocher(vp):
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


def vp_from_vs_brocher(vs):
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


def vs_from_vp_brocher(vp):
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


def v30gtl(vs30, vt, z, zt):
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
