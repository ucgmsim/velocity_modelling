from enum import Enum
import numpy as np
from typing import List, Dict
from pathlib import Path
from load_data import load_global_surface, load_1d_velo_sub_model
class BasinData:
    def __init__(self, nBasins: int):
        self.surf = [None]*nBasins # 2-D list of basin surfaces (nBasins x nBasinSurfaces)
        self.boundary = [None]*nBasins #  3-D list of basin boundaries (nBasins x nBasinBoundaries x 2 (lon, lat))

        self.basin_submodel_data = [None]*nBasins # 1-D list of basin submodel data (nBasins)
        self.perturbation_data = [None]*nBasins # 1-D list of perturbation data (nBasins), each element is a TomographyData object


class BasinSurfaceRead:
    def __init__(self, nLat, nLon):
        self.nLat = nLat
        self.nLon = nLon
        self.lati = np.zeros(nLat)
        self.loni = np.zeros(nLon)
        self.raster = np.zeros((nLon, nLat))
        self.maxLat = None
        self.minLat = None
        self.maxLon = None
        self.minLon = None


class GlobalMesh:
    def __init__(self, nX, nY, nZ):

        self.Lon = np.zeros((nX, nY))
        self.Lat = np.zeros((nX, nY))
        self.maxLat = 0.0
        self.minLat = 0.0
        self.maxLon = 0.0
        self.minLon = 0.0
        self.nX = nX
        self.nY = nY
        self.nZ = nZ
        self.X = np.zeros(nX)
        self.Y = np.zeros(nY)
        self.Z = np.zeros(nZ)

class PartialGlobalMesh:
    def __init__(self, nX, nZ):
        self.Lon = np.zeros(nX)
        self.Lat = np.zeros(nX)
        self.X = np.zeros(nX)
        self.Z = np.zeros(nZ)
        self.nX = nX
        self.nY = 1
        self.nZ = nZ
        self.Y = 0.0

class GlobalModelParameters:
    def __init__(self):
        self.nSurf: int = 0
        self.nVeloSubMod: int = 0
        self.surfaces: List[str] = []
        self.globalSurfFilenames: List[str] = []
        self.velo_submodels: List[str] = []
        self.veloMod1dFileName: List[str] = []
        self.tomographyName: str = ""
        self.nBasins: int = 0
        self.GTL: int = 0
        self.basins: List[str] = []
        self.nBasinSurfaces: List[int] = []
        self.nBasinBoundaries: List[int] = []
        self.basinBoundaryFilenames: List[List[str]] = []
        self.basinSurfaceNames: List[List[str]] = []
        self.basinSurfaceFilenames: List[List[str]] = []
        self.basinBoundaryNumber: List[List[int]] = []
        self.basinSubModelNames: List[List[str]] = []
        self.basin_edge_smoothing: bool = False


class GlobalSurfaces:
    def __init__(self):
        self.surf = []

class GlobalSurfaceRead:
    def __init__(self, nLat, nLon):
        self.nLat = nLat
        self.nLon = nLon
        self.lati = np.zeros(nLat)
        self.loni = np.zeros(nLon)
        self.raster = np.zeros((nLon, nLat))
        self.maxLat = None
        self.minLat = None
        self.maxLon = None
        self.minLon = None

class ModelExtent:
    def __init__(self, vm_params: Dict):
        self.originLat = vm_params['MODEL_LAT']
        self.originLon = vm_params['MODEL_LON']
        self.originRot = vm_params['MODEL_ROT'] # in degrees
        self.Xmax = vm_params['extent_x']
        self.Ymax = vm_params['extent_y']
        self.Zmax = vm_params['extent_zmax']
        self.Zmin = vm_params['extent_zmin']
        self.hDep = vm_params['hh']
        self.hLatLon = vm_params['hh']
        self.nx = vm_params['nx']
        self.ny = vm_params['ny']
        self.nz = vm_params['nz']

class SmoothingBoundary:
    def __init__(self):
        self.n = 0
        self.xPts = []
        self.yPts = []



class TomographyData:
    def __init__(self, elev: List[float], vs30_path: Path, special_offshore_tapering: bool, surf_tomo_path: Path, offshore_surface_path: Path, offshore_v1d_path: Path):
        self.surfDeps = elev
        self.vs30 = load_global_surface(vs30_path)
        self.special_offshore_tapering = special_offshore_tapering
        self.surf = []
        self.smooth_boundary = None

        for i in range(len(elev)):
            self.surf.append({}) # self.surf[i] is an empty dictionary
            if elev[i] == int(elev[i]): # if the elevation is an integer
                elev_name = f"{elev[i]}"
            else:
                elev_name = f"{elev[i]:.2f}".replace(".", "p")

            for vtype in VTYPE:
                tomofile = surf_tomo_path / f"surf_tomography_{vtype.name}_elev{elev_name}.in"
                self.surf[i][vtype.name]= load_global_surface(tomofile)

        self.offshore_distance_surface = load_global_surface(offshore_surface_path)
        self.offshore_basin_model_1d = load_1d_velo_sub_model(offshore_v1d_path)
        self.tomography_loaded = True

class VeloMod1DData:
    def __init__(self):
        self.Vp = []
        self.Vs = []
        self.Rho = []
        self.Dep = []
        self.nDep = 0

class VTYPE(Enum):
    vp = 0
    vs = 1
    rho = 2