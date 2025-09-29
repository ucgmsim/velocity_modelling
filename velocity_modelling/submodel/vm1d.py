# velocity_modelling/submodel/vm1d.py
import numpy as np
from .base import Submodel


class VM1DSubmodel(Submodel):
    def __init__(self, parameters: dict = None):
        self.parameters = parameters or {}

    def calculate(self, z_indices, depths, qualities_vector, vm1d_data=None, **kwargs):
        """Apply 1D velocity model."""
        model_depths = np.array(vm1d_data.bottom_depth) * -1000

        for i, depth in enumerate(depths):
            layer_idx = np.searchsorted(-model_depths, -depth)
            if 0 <= layer_idx < len(model_depths):
                qualities_vector.vp[z_indices[i]] = vm1d_data.vp[layer_idx]
                qualities_vector.vs[z_indices[i]] = vm1d_data.vs[layer_idx]
                qualities_vector.rho[z_indices[i]] = vm1d_data.rho[layer_idx]