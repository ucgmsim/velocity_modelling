# velocity_modelling/submodel/constant.py
from .base import Submodel


class ConstantSubmodel(Submodel):
    def __init__(self, parameters: dict):
        self.vp = parameters['vp']
        self.vs = parameters['vs']
        self.rho = parameters['rho']

    def calculate(self, z_indices, depths, qualities_vector, **kwargs):
        """Apply constant values."""
        qualities_vector.vp[z_indices] = self.vp
        qualities_vector.vs[z_indices] = self.vs
        qualities_vector.rho[z_indices] = self.rho