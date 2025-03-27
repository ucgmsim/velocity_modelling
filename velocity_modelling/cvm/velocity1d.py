"""
1D Velocity Model Module.

This module provides the VelocityModel1D class for representing one-dimensional
velocity-depth profiles. These profiles define seismic properties (P-wave velocity,
S-wave velocity, density and quality factors (Qp, Qs)) as a function of depth.

The 1D velocity models serve as reference profiles for 3D velocity modeling and
are used as baseline models in regions where more detailed information is unavailable.
"""

import numpy as np


class VelocityModel1D:
    """
    Class representing a one-dimensional velocity-depth profile.

    This class stores arrays of P-wave velocity, S-wave velocity, density, quality factors (Qp, Qs) and
    corresponding depths, allowing for the representation of layered earth models.

    Parameters
    ----------
    vp : np.ndarray
        P-wave velocities (km/s).
    vs : np.ndarray
        S-wave velocities (km/s).
    rho : np.ndarray
        Densities (g/cm³).
    qp : np.ndarray
        P-wave quality factors.
    qs : np.ndarray
        S-wave quality factors.
    depth : np.ndarray
        Depths (m) of the bottom of each layer

    Attributes
    ----------
    vp : np.ndarray
        P-wave velocities (km/s).
    vs : np.ndarray
        S-wave velocities (km/s).
    rho : np.ndarray
        Densities (g/cm³).
    qp : np.ndarray
        P-wave quality factors.
    qs : np.ndarray
        S-wave quality factors.
    depth : np.ndarray
        Depths (m) of the bottom of each layer
    n_depth : int
        Number of depth points.
    """

    def __init__(
        self,
        vp: np.ndarray,
        vs: np.ndarray,
        rho: np.ndarray,
        qp: np.ndarray,
        qs: np.ndarray,
        depth: np.ndarray,
    ):
        """
        Initialize the VelocityModel1D.

        """
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.depth = depth
        self.n_depth = len(depth)
        if not (len(vp) == len(vs) == len(rho) == len(qp) == len(qs) == len(depth)):
            raise ValueError("Input arrays must have the same length")
