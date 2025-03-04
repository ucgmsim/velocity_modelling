import numpy as np


class VelocityModel1D:
    def __init__(
        self, vp: np.ndarray, vs: np.ndarray, rho: np.ndarray, depth: np.ndarray
    ):
        """
        Initialize the VelocityModel1D.

        Parameters
        ----------
        vp : np.ndarray
            P-wave velocity.
        vs : np.ndarray
            S-wave velocity.
        rho : np.ndarray
            Density.
        depth : np.ndarray
            Depth.

        """
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.depth = depth
        self.n_depth = len(vp)  # maybe should be len(dep) but I'm not sure
        assert len(vp) == len(vs) == len(rho) == len(depth)
