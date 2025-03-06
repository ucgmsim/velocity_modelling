"""
1D Velocity Model Module.

This module provides the VelocityModel1D class for representing one-dimensional
velocity-depth profiles. These profiles define seismic properties (P-wave velocity,
S-wave velocity, and density) as a function of depth.

The 1D velocity models serve as reference profiles for 3D velocity modeling and
are used as baseline models in regions where more detailed information is unavailable.
"""

import numpy as np


class VelocityModel1D:
    """
    Class representing a one-dimensional velocity-depth profile.

    This class stores arrays of P-wave velocity, S-wave velocity, density, and
    corresponding depths, allowing for the representation of layered earth models.

    Attributes
    ----------
    vp : np.ndarray
        P-wave velocities (km/s).
    vs : np.ndarray
        S-wave velocities (km/s).
    rho : np.ndarray
        Densities (g/cm³).
    depth : np.ndarray
        Depths (m).
    n_depth : int
        Number of depth points.
    """

    def __init__(
        self, vp: np.ndarray, vs: np.ndarray, rho: np.ndarray, depth: np.ndarray
    ):
        """
        Initialize the VelocityModel1D.

        Parameters
        ----------
        vp : np.ndarray
            P-wave velocities (km/s).
        vs : np.ndarray
            S-wave velocities (km/s).
        rho : np.ndarray
            Densities (g/cm³).
        depth : np.ndarray
            Depths (m).
        """
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.depth = depth
        self.n_depth = len(depth)
        assert (
            len(vp) == len(vs) == len(rho) == len(depth)
        ), "Input arrays must have the same length"
