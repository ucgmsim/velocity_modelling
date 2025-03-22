"""
Constants for the velocity modelling package.

"""

from enum import Enum
from pathlib import Path

MAX_DIST_SMOOTH = 10  # distance in KM to smooth tomography over

MAX_LAT_SURFACE_EXTENSION = 10  # value in degrees the global (Vs30, DEM, tomography) surface files may be extended by
MAX_LON_SURFACE_EXTENSION = 10  # value in degrees the global (Vs30, DEM, tomography) surface files may be extended by
EARTH_RADIUS_MEAN = 6378.139

CVM_ROOT = Path(__file__).parent
DATA_ROOT = (
    CVM_ROOT / "data"
)  # default value, can be overridden with --data-root argument
MODEL_VERSIONS_ROOT = CVM_ROOT / "model_versions"
NZVM_REGISTRY_PATH = CVM_ROOT / "nzvm_registry.yaml"

DEFAULT_OFFSHORE_1D_MODEL = "canterbury1d_v2"  # vm1d name for offshore 1D model
DEFAULT_OFFSHORE_DISTANCE = "offshore"  # surface name for offshore distance
LON_GRID_DIM_MAX = 10260
LAT_GRID_DIM_MAX = 19010
DEP_GRID_DIM_MAX = 4500
FLAT_CONST = 298.256
ERAD = 6378.139  # Earth's radius in km
RPERD = 0.017453292


class VelocityTypes(Enum):
    """
    Enum for the velocity type.

    0: P-wave velocity
    1: S-wave velocity
    2: Density
    """

    vp = 0
    vs = 1
    rho = 2


class TopoTypes(Enum):
    """
    Enum for the topography types.

    0: BULLDOZED
    1: SQUASHED
    2: SQUASHED_TAPERED
    3: TRUE
    """

    BULLDOZED = 0
    SQUASHED = 1
    SQUASHED_TAPERED = 2
    TRUE = 3


class WriteFormat(Enum):
    """
    Enum for the write format types.

    0: EMOD3D
    1: CSV

    """

    EMOD3D = 0
    CSV = 1
