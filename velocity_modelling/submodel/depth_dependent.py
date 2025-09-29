# velocity_modelling/submodel/depth_dependent.py
import numpy as np
from .base import Submodel

class DepthDependentSubmodel(Submodel):
    def __init__(self, parameters: dict):
        self.shallow = parameters['shallow']
        self.deep = parameters['deep']
        self.transition_depth = parameters.get('transition_depth', 100)

    def calculate(self, z_indices, depths, qualities_vector,
                 basin_surface_depths=None, **kwargs):
        """Apply depth-dependent values with transition."""
        for i, depth in enumerate(depths):
            # Calculate depth from surface
            relative_depth = basin_surface_depths[0] - depth

            if relative_depth < self.transition_depth:
                # Linear transition
                weight = relative_depth / self.transition_depth
                qualities_vector.vp[z_indices[i]] = (
                    self.shallow['vp'] * (1-weight) + self.deep['vp'] * weight
                )
                qualities_vector.vs[z_indices[i]] = (
                    self.shallow['vs'] * (1-weight) + self.deep['vs'] * weight
                )
                qualities_vector.rho[z_indices[i]] = (
                    self.shallow['rho'] * (1-weight) + self.deep['rho'] * weight
                )
            else:
                qualities_vector.vp[z_indices[i]] = self.deep['vp']
                qualities_vector.vs[z_indices[i]] = self.deep['vs']
                qualities_vector.rho[z_indices[i]] = self.deep['rho']