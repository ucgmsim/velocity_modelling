"""
Constants for the velocity modelling package.

"""

from enum import Enum, auto
from pathlib import Path

MAX_DIST_SMOOTH = 10  # distance in KM to smooth tomography over

MAX_LAT_SURFACE_EXTENSION = 10  # value in degrees the global (Vs30, DEM, tomography) surface files may be extended by
MAX_LON_SURFACE_EXTENSION = 10  # value in degrees the global (Vs30, DEM, tomography) surface files may be extended by
EARTH_RADIUS_MEAN = 6378.139

CVM_ROOT = Path(__file__).parent
DATA_ROOT = (
    CVM_ROOT / "nzcvm_data"
)  # default value, can be overridden with --data-root argument
MODEL_VERSIONS_ROOT = CVM_ROOT / "model_versions"
NZCVM_REGISTRY_PATH = DATA_ROOT / "nzcvm_registry.yaml"

DEFAULT_OFFSHORE_1D_MODEL = "canterbury1d_v2"  # vm1d name for offshore 1D model
DEFAULT_OFFSHORE_DISTANCE = (
    "global/surface/shoreline_distance_2k.h5"  # surface for offshore distance
)
LON_GRID_DIM_MAX = 10260  # Maximum number of grid points in longitude dimension
LAT_GRID_DIM_MAX = 19010  # Maximum number of grid points in latitude dimension
DEP_GRID_DIM_MAX = 4500  # Maximum number of grid points in depth dimension
FLAT_CONST = 298.256  # Earth's flattening constant (1/f)
ERAD = 6378.139  # Earth's radius in km
RPERD = 0.017453292  # Radians per degree (π/180) conversion factor


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
    2: HDF5
    """

    EMOD3D = auto()
    CSV = auto()
    HDF5 = auto()
