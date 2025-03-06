from enum import Enum
from pathlib import Path

MAX_DIST_SMOOTH = 10  # distance in KM to smooth tomography over

MAX_LAT_SURFACE_EXTENSION = 10  # value in degrees the global (Vs30, DEM, tomography) surface files may be extended by
MAX_LON_SURFACE_EXTENSION = 10  # value in degrees the global (Vs30, DEM, tomography) surface files may be extended by
EARTH_RADIUS_MEAN = 6378.139

DATA_ROOT = Path(__file__).parent.parent / "Data"
NZVM_REGISTRY_PATH = DATA_ROOT / "nzvm_registry.yaml"

DEFAULT_OFFSHORE_1D_MODEL = "Cant1D_v2"  # vm1d name for offshore 1D model
DEFAULT_OFFSHORE_DISTANCE = "offshore"  # surface name for offshore distance
LON_GRID_DIM_MAX = 10260
LAT_GRID_DIM_MAX = 19010
DEP_GRID_DIM_MAX = 4500
FLAT_CONST = 298.256
ERAD = 6378.139  # Earth's radius in km
RPERD = 0.017453292


class VTYPE(Enum):
    vp = 0
    vs = 1
    rho = 2
