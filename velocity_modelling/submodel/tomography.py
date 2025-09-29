# velocity_modelling/submodel/tomography.py
import numpy as np
from .base import Submodel


class TomographySubmodel(Submodel):
    def __init__(self, parameters: dict = None):
        self.parameters = parameters or {}

    def calculate(self, z_indices, depths, qualities_vector,
                  tomography_data=None, mesh_vector=None, **kwargs):
        """Interpolate from tomography surfaces."""
        # Find bracketing surfaces
        surf_depths = np.array(tomography_data.surf_depth) * 1000
        indices = np.searchsorted(-surf_depths, -depths)

        # Interpolate between surfaces
        for i, depth in enumerate(depths):
            idx = indices[i]
            if idx == 0:
                # Above first surface
                qualities_vector.vp[z_indices[i]] = tomography_data.surfaces[0]['vp']
                qualities_vector.vs[z_indices[i]] = tomography_data.surfaces[0]['vs']
                qualities_vector.rho[z_indices[i]] = tomography_data.surfaces[0]['rho']
            elif idx >= len(surf_depths):
                # Below last surface
                qualities_vector.vp[z_indices[i]] = tomography_data.surfaces[-1]['vp']
                qualities_vector.vs[z_indices[i]] = tomography_data.surfaces[-1]['vs']
                qualities_vector.rho[z_indices[i]] = tomography_data.surfaces[-1]['rho']
            else:
                # Interpolate
                weight = (depth - surf_depths[idx - 1]) / (surf_depths[idx] - surf_depths[idx - 1])
                qualities_vector.vp[z_indices[i]] = (
                        tomography_data.surfaces[idx - 1]['vp'] * (1 - weight) +
                        tomography_data.surfaces[idx]['vp'] * weight
                )
                # Similar for vs and rho