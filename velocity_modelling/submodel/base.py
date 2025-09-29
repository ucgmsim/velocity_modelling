# velocity_modelling/submodel/base.py
from abc import ABC, abstractmethod
import importlib
import yaml
from pathlib import Path
import numpy as np


class Submodel(ABC):
    """Base class for all velocity submodels."""

    @abstractmethod
    def calculate(self, z_indices: np.ndarray, depths: np.ndarray,
                  qualities_vector, **context) -> None:
        """Calculate vp, vs, rho for given depths."""
        pass


class SubmodelLoader:
    """Simple loader for submodels."""

    _registry = None
    _instances = {}

    @classmethod
    def get(cls, submodel_type: str, parameters: dict = None):
        """Get or create a submodel instance."""
        # Load registry once
        if cls._registry is None:
            registry_path = Path(__file__).parent.parent / "submodel_registry.yaml"
            with open(registry_path, 'r') as f:
                cls._registry = yaml.safe_load(f)['submodels']

        # Create unique key for caching
        cache_key = f"{submodel_type}_{parameters}"

        if cache_key not in cls._instances:
            if submodel_type not in cls._registry:
                raise ValueError(f"Unknown submodel type: {submodel_type}")

            config = cls._registry[submodel_type]
            module = importlib.import_module(config['module'])
            submodel_class = getattr(module, config['class'])
            cls._instances[cache_key] = submodel_class(parameters)

        return cls._instances[cache_key]