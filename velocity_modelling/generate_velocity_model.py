import concurrent.futures
from enum import Enum
from logging import Logger
import logging

import os
from pathlib import Path
from typing import List, Dict
import argparse
import yaml
import numpy as np
import sys

LON_GRID_DIM_MAX = 10260
LAT_GRID_DIM_MAX = 19010
DEP_GRID_DIM_MAX = 4500

# constants for coord generation functions
FLAT_CONST = 298.256
ERAD  = 6378.139 # Earth's radius in km
RPERD = 0.017453292

from typing import List, Dict

DATA_ROOT = Path(__file__).parent / "Data"

nzvm_registry_path = DATA_ROOT/'nzvm_registry.yaml'

DEFAULT_OFFSHORE_1D_MODEL = "v1DsubMod_v2" # vm1d name for offshore 1D model
DEFAULT_OFFSHORE_DISTANCE = "offshore"  # surface name for offshore distance

class BasinData:
    def __init__(self):
        self.surf = [] # 2-D list of basin surfaces (nBasins x nBasinSurfaces)
        self.boundary_num_points = [] # 2-D list of number of points in each basin boundary (nBasins x nBasinBoundaries)
        self.boundary_lat = [] # 3-D list of basin boundary latitudes (nBasins x nBasinBoundaries x size of boundary)
        self.boundary_lon = [] # 3-D list of basin boundary longitudes (nBasins x nBasinBoundaries x size of boundary)
        self.min_lon_boundary = [] # 2-D list of minimum longitude of basin boundary (nBasins x nBasinBoundaries)
        self.max_lon_boundary = [] # 2-D list of maximum longitude of basin boundary (nBasins x nBasinBoundaries)
        self.min_lat_boundary = [] # 2-D list of minimum latitude of basin boundary (nBasins x nBasinBoundaries)
        self.max_lat_boundary = [] # 2-D list of maximum latitude of basin boundary (nBasins x nBasinBoundaries)

        self.basin_submodel_data = [] # 1-D list of basin submodel data (nBasins)
        self.perturbation_data = [] # 1-D list of perturbation data (nBasins), each element is a TomographyData object


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


def write_velo_mod_corners_text_file(global_mesh: GlobalMesh, output_dir: str,  logger: Logger) -> None:
    """
    Write velocity model corners to a text file.

    Parameters
    ----------
    global_mesh : GlobalMesh
        An object containing the global mesh data, including longitude and latitude arrays.
    output_dir : str
        Directory where the output log file will be saved.

    Returns
    -------
    None
    """
    log_file_name = Path(output_dir) / 'Log' / 'VeloModCorners.txt'
    log_file_name.parent.mkdir(parents=True, exist_ok=True)

    with log_file_name.open('w') as fp:
        fp.write(">Velocity model corners.\n")
        fp.write(">Lon\tLat\n")
        fp.write(f"{global_mesh.Lon[0][global_mesh.nY - 1]}\t{global_mesh.Lat[0][global_mesh.nY - 1]}\n")
        fp.write(f"{global_mesh.Lon[0][0]}\t{global_mesh.Lat[0][0]}\n")
        fp.write(f"{global_mesh.Lon[global_mesh.nX - 1][0]}\t{global_mesh.Lat[global_mesh.nX - 1][0]}\n")
        fp.write(
            f"{global_mesh.Lon[global_mesh.nX - 1][global_mesh.nY - 1]}\t{global_mesh.Lat[global_mesh.nX - 1][global_mesh.nY - 1]}\n")

    logger.info("Velocity model corners file write complete.")
def great_circle_projection(x: np.ndarray, y: np.ndarray,  amat: np.ndarray, erad: float=ERAD, g0: float=0, b0: float=0) -> tuple[float, float]:
    """
    Project x, y coordinates to geographic coordinates (longitude, latitude) using a great circle projection.

    Parameters
    ----------
    x : float
        X-coordinate.
    y : float
        Y-coordinate.

    erad : float
        Earth's radius.

    amat : np.ndarray
        Transformation matrix.
    ainv : np.ndarray
        Inverse transformation matrix.

    Returns
    -------
    tuple[float, float]
        Computed latitude and longitude.
    """

    cosB = np.cos(x/erad - b0)
    sinB = np.sin(x/erad - b0)

    cosG = np.cos(y/erad - g0)
    sinG = np.sin(y/erad - g0)

    xp = sinG*cosB*np.sqrt(1+sinB*sinB*sinG*sinG)
    yp = sinB*cosG*np.sqrt(1+sinB*sinB*sinG*sinG)
    zp = np.sqrt(1-xp*xp - yp*yp)
    # Stack xp, yp, zp along the last axis to create a 3D array
    coords = np.stack((xp, yp, zp), axis=0)  # Shape: (3, 723, 726)

    # Perform matrix multiplication
    xg, yg, zg = np.tensordot(amat, coords, axes=([1], [0]))  # Shape: (723, 726, 3)

    lat = np.where(
        np.isclose(zg, 0),
        0,
        90 - np.arctan(np.sqrt(xg ** 2 + yg ** 2) / zg) / RPERD - np.where(zg < 0, 180, 0)
    )

    lon = np.where(
        np.isclose(xg, 0),
        0,
        np.arctan(yg / xg) / RPERD - np.where(xg < 0, 180, 0)
    )
    lon = lon % 360
    return lat, lon

def gen_full_model_grid_great_circle(model_extent: ModelExtent, logger: Logger) -> GlobalMesh:
    """
    Generate the grid of latitude, longitude, and depth points using the point radial distance method.

    Parameters
    ----------
    model_extent : ModelExtent
        Object containing the extent, spacing, and version of the model.

    Returns
    -------
    global_mesh: GlobalMesh
        Object containing the generated grid of latitude, longitude, and depth points.
    """


    nX = int (np.round(model_extent.Xmax / model_extent.hLatLon))
    nY = int (np.round(model_extent.Ymax / model_extent.hLatLon))
    nZ = int(np.round((model_extent.Zmax - model_extent.Zmin) / model_extent.hDep))

    global_mesh = GlobalMesh(nX,nY,nZ)

    global_mesh.maxLat = -180
    global_mesh.minLat = 0
    global_mesh.maxLon = 0
    global_mesh.minLon = 180

    assert global_mesh.nX == model_extent.nx
    assert global_mesh.nY == model_extent.ny
    assert global_mesh.nZ == model_extent.nz

    if any([
        global_mesh.nX >= LON_GRID_DIM_MAX,
        global_mesh.nY >= LAT_GRID_DIM_MAX,
        global_mesh.nZ >= DEP_GRID_DIM_MAX
    ]):
        raise ValueError(f"Grid dimensions exceed maximum allowable values. X={LON_GRID_DIM_MAX}, Y={LAT_GRID_DIM_MAX}, Z={DEP_GRID_DIM_MAX}")

    if global_mesh.nZ != 1:
        logger.info(f"Number of model points. nx: {global_mesh.nX}, ny: {global_mesh.nY}, nz: {global_mesh.nZ}.")


    for i in range(global_mesh.nX):
        global_mesh.X[i] = 0.5 * model_extent.hLatLon + model_extent.hLatLon * i - 0.5 * model_extent.Xmax

    for i in range(global_mesh.nY):
        global_mesh.Y[i] = 0.5 * model_extent.hLatLon + model_extent.hLatLon * i - 0.5 * model_extent.Ymax

    for i in range(global_mesh.nZ):
        global_mesh.Z[i] = -1000 * (model_extent.Zmin + model_extent.hDep * (i + 0.5))

    arg = model_extent.originRot * RPERD
    cosA = np.cos(arg)
    sinA = np.sin(arg)

    arg = (90.0 - model_extent.originLat) * RPERD
    cosT = np.cos(arg)
    sinT = np.sin(arg)

    arg = model_extent.originLon * RPERD
    cosP = np.cos(arg)
    sinP = np.sin(arg)

    amat = np.array([
        cosA * cosT * cosP + sinA * sinP,
        sinA * cosT * cosP - cosA * sinP,
        sinT * cosP,
        cosA * cosT * sinP - sinA * cosP,
        sinA * cosT * sinP + cosA * cosP,
        sinT * sinP,
        -cosA * sinT,
        -sinA * sinT,
        cosT
    ]).reshape((3, 3))

    det = np.linalg.det(amat)
    ainv = np.linalg.inv(amat) / det

    g0 = 0.0
    b0 = 0.0

    # for iy in range(global_mesh.nY):
    #     for ix in range(global_mesh.nX):
    #         x = global_mesh.X[ix]
    #         y = global_mesh.Y[iy]
    #         lat, lon = great_circle_projection(x, y, ERAD, g0, b0, amat)
    #         global_mesh.Lon[ix][iy] = lon
    #         global_mesh.Lat[ix][iy] = lat

    X, Y = np.meshgrid(global_mesh.X[:global_mesh.nX], global_mesh.Y[:global_mesh.nY], indexing='ij')
    lat_lon =  great_circle_projection(X, Y, amat)
    global_mesh.Lat[:global_mesh.nX, :global_mesh.nY], global_mesh.Lon[:global_mesh.nX, :global_mesh.nY] = lat_lon

    global_mesh.maxLat = np.max(global_mesh.Lat)
    global_mesh.maxLon = np.max(global_mesh.Lon)
    global_mesh.minLat = np.min(global_mesh.Lat)
    global_mesh.minLon = np.min(global_mesh.Lon)

    logger.info("Completed Generation of Model Grid.")
    return global_mesh

#SAMPLE vm_params file

#

def nzvm_registry_get_vm(version):
    vminfo_list = nzvm_registry.get('vm')
    for vminfo in vminfo_list:
        if str(vminfo['version']) == version:
            return vminfo
    return None

def nzvm_registry_get_basin(basin_name):
    basin_list = nzvm_registry.get('basin')
    for basin in basin_list:
        if basin['name'] == basin_name:
            return basin
    return None

def nzvm_registry_get_surface(surf_name):
    surf_list = nzvm_registry.get('surface')
    for surf in surf_list:
        if surf['name'] == surf_name:
            return surf
    return None

def nzvm_registry_get_tomography(tomo_name):
    tomo_list = nzvm_registry.get('tomography')
    for tomo in tomo_list:
        if tomo['name'] == tomo_name:
            return tomo
    return None

def nzvm_registry_get_vm1d(vm1d_name):
    vm1d_list = nzvm_registry.get('vm1d')
    for vm1d in vm1d_list:
        if vm1d['name'] == vm1d_name:
            return vm1d
    return None

def load_1d_velo_sub_model(v1d_path: Path) -> VeloMod1DData:
    """
    Load a 1D velocity submodel into memory.

    Parameters
    ----------
    file_name : str
        The filename to open and read.

    Returns
    -------
    VeloMod1DData
        Struct containing a 1D velocity model.
    """

    velo_mod_1d_data = VeloMod1DData()
    try:
        with open(v1d_path, "r") as file:
            # Discard header line
            next(file)
            data = np.loadtxt(file)
            velo_mod_1d_data.Vp = data[:, 0].tolist()
            velo_mod_1d_data.Vs = data[:, 1].tolist()
            velo_mod_1d_data.Rho = data[:, 2].tolist()
            velo_mod_1d_data.Dep = data[:, 5].tolist()
            velo_mod_1d_data.nDep = len(velo_mod_1d_data.Dep)
    except FileNotFoundError:
        print(f"Error 1D velocity model file {v1d_path} not found.")
        exit(1)

    return velo_mod_1d_data

def load_basin_data(basin_names: List[str]):
    """
    Purpose: load all basin data into the basin_data structure

    Input variables:
    basin_data - dict containing basin data (surfaces submodels etc)
    global_model_parameters - dict containing all model parameters (surface names, submodel names, basin names etc)

    Output variables:
    n.a.
    """
    basins = [nzvm_registry_get_basin(name) for name in basin_names]

    basin_data = BasinData()

    basin_data.surf = [None] * len(basins) # 2-D list of basin surfaces (nBasins x nBasinSurfaces)
    # loop over nBasins and load in surfaces, boundaries and sub-models
    for i, basin in enumerate(basins):
        basin_data.surf[i] = [load_basin_surface( DATA_ROOT / surface['path']) for surface in basin['surfaces']]

        load_basin_boundaries(i, basin_data, basin)
        load_basin_sub_model_data(i, basin_data, basin)

    print("All basin surfaces loaded.")
    print("All basin boundaries loaded.")
    print("All basin sub model data loaded.")

def load_all_basin_surfaces(basin: dict):
    """
    Purpose: load all basin surfaces into the basin_data structure

    Input variables:
    basin - dict containing basin data

    Output variables:
    n.a.
    """
    basin_surfaces = basin['surfaces']
    surfs = []

    for surface in basin_surfaces:
        surf_path = DATA_ROOT / surface['path']
        surfs.append(load_basin_surface(surf_path))
    return surfs

def load_basin_boundaries(basin_num: int, basin_data: BasinData, basin: dict) -> None:
    """
    Load all basin boundaries.

    Parameters
    ----------
    basin_num : int
        The basin number pertaining to the basin of interest.
    basin_data : BasinData
        Struct containing basin data (surfaces, submodels, etc).
    global_model_parameters : GlobalModelParameters
        Struct containing all model parameters (surface names, submodel names, basin names, etc).

    Returns
    -------
    None
    """
    for i, boundary in enumerate(basin['boundaries']):
        file_path = boundary['path']
        try:
            data = np.loadtxt(file_path)
            lon = data[:, 0]
            lat = data[:, 1]
            basin_data.boundary_lon[basin_num][i][:len(lon)] = lon
            basin_data.boundary_lat[basin_num][i][:len(lat)] = lat
            basin_data.boundary_num_points[basin_num][i] = len(lon)
            basin_data.min_lon_boundary[basin_num][i] = np.min(lon)
            basin_data.max_lon_boundary[basin_num][i] = np.max(lon)
            basin_data.min_lat_boundary[basin_num][i] = np.min(lat)
            basin_data.max_lat_boundary[basin_num][i] = np.max(lat)

            assert lon[-1] == lon[0]
            assert lat[-1] == lat[0]
        except FileNotFoundError:
            print(f"Error basin boundary file {file_path} not found.")
            exit(1)
        except Exception as e:
            print(f"Error reading basin boundary file {file_path}: {e}")
            exit(1)

def load_basin_surface(basin_surface_path:Path):
    try:
        with open(basin_surface_path, 'r') as f:
            nLat, nLon = map(int, f.readline().split())
            basin_surf_read = BasinSurfaceRead(nLat, nLon)

            # Reading latitudes and longitudes in one go
            latitudes = np.fromfile(f, dtype=float, count=nLat, sep=' ')
            longitudes = np.fromfile(f, dtype=float, count=nLon, sep=' ')

            basin_surf_read.lati = latitudes
            basin_surf_read.loni = longitudes

            # Reading raster data efficiently
            raster_data = np.fromfile(f, dtype=float, count=nLat*nLon, sep=' ')
            basin_surf_read.raster = raster_data.reshape((nLon, nLat)).T

            firstLat = basin_surf_read.lati[0]
            lastLat = basin_surf_read.lati[nLat - 1]
            basin_surf_read.maxLat = max(firstLat, lastLat)
            basin_surf_read.minLat = min(firstLat, lastLat)

            firstLon = basin_surf_read.loni[0]
            lastLon = basin_surf_read.loni[nLon - 1]
            basin_surf_read.maxLon = max(firstLon, lastLon)
            basin_surf_read.minLon = min(firstLon, lastLon)

            return basin_surf_read

    except FileNotFoundError:
        print(f"Error basin surface file {basin_surface_path} not found.")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def load_tomo_surface_data(tomo_name: str, offshore_surface_name: str = DEFAULT_OFFSHORE_DISTANCE, offshore_v1d_name: str = DEFAULT_OFFSHORE_1D_MODEL) -> VeloMod1DData:

    tomo = nzvm_registry_get_tomography(tomo_name)
    offshore_surface= nzvm_registry_get_surface(offshore_surface_name)
    offshore_v1d = nzvm_registry_get_vm1d(offshore_v1d_name)

    return TomographyData(tomo['elev'], DATA_ROOT/tomo['vs30_path'], tomo['special_offshore_tapering'], DATA_ROOT/tomo['path'], DATA_ROOT/offshore_surface['path'], DATA_ROOT/offshore_v1d['path'])

def load_all_global_data(global_model_params: dict, logger: Logger) : #-> (VeloMod1DData, NZTomographyData, GlobalSurfaces, BasinData):

    """
    Load all data required to generate the velocity model, global surfaces, sub velocity models, and all basin data.

    Parameters
    ----------
    global_model_params : dict
        Struct containing all model parameters (surface names, submodel names, basin names, etc).
    calculation_log : CalculationLog
        Struct containing calculation data and output directory (tracks various parameters for error reporting, etc).
    velo_mod_1d_data : VeloMod1DData
        Struct containing a 1D velocity model.
    nz_tomography_data : NZTomographyData
        Struct containing tomography sub velocity model data (tomography surfaces depths, etc).
    global_surfaces : GlobalSurfaces
        Struct containing pointers to global surfaces (whole domain surfaces which sub velocity models apply between).
    basin_data : BasinData
        Struct containing basin data (surfaces, submodels, etc).

    Returns
    -------
    VeloMod1DData

    NZTomographyData

    GlobalSurfaces

    BasinData
    """
    velo_mod_1d_data = None
    nz_tomography_data = None

    # read in sub velocity models
    print("Loading global velocity submodel data.")
    for i in range(len(global_model_params['velo_submodels'])):
        if global_model_params['velo_submodels'][i] == "v1DsubMod":
            velo_mod_1d_data=load_1d_velo_sub_model( global_model_params['velo_submodels'][i])
            print("Loaded 1D velocity model data.")
        elif global_model_params['velo_submodels'][i] == "NaNsubMod":
            # no data required for NaN velocity sub model, leave as placeholder
            pass
        else: # tomography sub model EPtomo2010subMod eg. 2020_NZ_OFFSHORE

            nz_tomography_data = load_tomo_surface_data(global_model_params['tomography'])
            print("Loaded tomography data.")

    pass

    # load in vector containing basin 'wall-type' boundaries to apply smoothing near
    if nz_tomography_data is not None:
        nz_tomography_data.smooth_boundary = SmoothingBoundary()
        load_smooth_boundaries(nz_tomography_data, global_model_params['basins'])

    print("Completed loading of global velocity submodel data.")

    # read in global surfaces
    global_surfaces = load_global_surface_data(global_model_params['surfaces'])
    print("Completed loading of global surfaces.")
    #
    # read in basin surfaces and boundaries
    print("Loading basin data.")
    basin_data = load_basin_data(global_model_params['basins'])
    print("Completed loading basin data.")
    print("All global data loaded.")
    return velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data

def load_global_surface(surface_file: Path):
    try:    
        with open(surface_file, 'r') as f:
            nLat, nLon = map(int, f.readline().split())
            global_surf_read = GlobalSurfaceRead(nLat, nLon)
            
#            for i in range(nLat):
#                global_surf_read.lati[i] = float(f.readline().strip())
#                
#            for i in range(nLon):
#                global_surf_read.loni[i] = float(f.readline().strip())
#                
#            for i in range(nLat):
#                for j in range(nLon):
#                    global_surf_read.raster[j][i] = float(file.readline().strip())
           
            # Reading latitudes and longitudes in one go
            latitudes = np.fromfile(f, dtype=float, count=nLat, sep=' ')
            longitudes = np.fromfile(f, dtype=float, count=nLon, sep=' ')

            global_surf_read.lati = latitudes
            global_surf_read.loni = longitudes

            # Reading raster data efficiently
            raster_data = np.fromfile(f, dtype=float, count=nLat*nLon, sep=' ')
            try:
                global_surf_read.raster = raster_data.reshape((nLon, nLat)).T
            except:
                sys.exit(f"Error: raster data shape {raster_data.shape} does not match lat/lon shape {nLat}x{nLon}")

            firstLat = global_surf_read.lati[0]
            lastLat = global_surf_read.lati[nLat - 1]
            global_surf_read.maxLat = max(firstLat, lastLat)
            global_surf_read.minLat = min(firstLat, lastLat)
            
            firstLon = global_surf_read.loni[0]
            lastLon = global_surf_read.loni[nLon - 1]
            global_surf_read.maxLon = max(firstLon, lastLon)
            global_surf_read.minLon = min(firstLon, lastLon)
            
            return global_surf_read
        
    except FileNotFoundError:
        print(f"Error surface file {surface_file} not found.")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

def load_global_surface_data(global_surface_names: List[str]):
    surfaces = [nzvm_registry_get_surface(name) for name in global_surface_names]
    global_surfaces = GlobalSurfaces()
    
    for surface in surfaces:
        global_surfaces.surf.append(load_global_surface(DATA_ROOT/surface['path']))

    return global_surfaces
# def generate_velocity_model(global_mesh, model_extent, global_model_parameters, velo_mod_1d_data, nz_tomography_data,
#                             global_surfaces, basin_data, calculation_log, gen_extract_velo_mod_call, output_dir,
#                             smoothing_required, n_pts_smooth):


def load_smooth_boundaries(nz_tomography_data: TomographyData, basin_names: List[str]):
    smooth_bound = nz_tomography_data.smooth_boundary
    count = 0

    for basin_name in basin_names:
        basin = nzvm_registry_get_basin(basin_name)
        if "smoothing" in basin:
            boundary_vec_filename = DATA_ROOT / basin["smoothing"]["path"]

            if boundary_vec_filename.exists():
                print(f"Loading offshore smoothing file: {boundary_vec_filename}.")
                try:
                    data = np.fromfile(boundary_vec_filename, dtype=float, sep=' ')
                    x_pts = data[0::2]
                    y_pts = data[1::2]
                    smooth_bound.xPts.extend(x_pts)
                    smooth_bound.yPts.extend(y_pts)
                    count += len(x_pts)
                except Exception as e:
                    print(f"Error reading smoothing boundary vector file {boundary_vec_filename}: {e}")
            else:
                print(f"Error smoothing boundary vector file {boundary_vec_filename} not found.")
        else:
            print(f"Smoothing not required for basin {basin_name}.")
    # print(count)
    # assert count <= MAX_NUM_POINTS_SMOOTH_VEC
    smooth_bound.n = count


def generate_velocity_model(output_dir: str, vm_params: Dict, logger: Logger) -> None:
    """
    print("Generating velocity model")
    """
    model_extent = ModelExtent(vm_params)
    global_mesh = gen_full_model_grid_great_circle(model_extent,logger)
    write_velo_mod_corners_text_file(global_mesh, output_dir, logger)

    global_model_params=nzvm_registry_get_vm(vm_params['model_version'])

    velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data = load_all_global_data(global_model_params, logger)

    # for j in range(global_mesh.nY):
    #
    #     print(f"\rGenerating velocity model {j * 100 / global_mesh.nY:.2f}% complete.", end="")
    #     partial_global_mesh = extract_partial_mesh(global_mesh, j)
    #     partial_global_qualities = PartialGlobalQualities(partial_global_mesh.nX, partial_global_mesh.nZ)
    #
    #     for k in range(partial_global_mesh.nX):
    #         in_basin = InBasin()
    #         partial_global_surface_depths = PartialGlobalSurfaceDepths()
    #         partial_basin_surface_depths = PartialBasinSurfaceDepths()
    #         qualities_vector = QualitiesVector()
    #         extended_qualities_vector = QualitiesVector()
    #
    #         if smoothing_required == 1:
    #             extended_mesh_vector = extend_mesh_vector(partial_global_mesh, n_pts_smooth, model_extent.hDep * 1000,
    #                                                       k)
    #             assign_qualities(global_model_parameters, velo_mod_1d_data, nz_tomography_data, global_surfaces,
    #                              basin_data, extended_mesh_vector, partial_global_surface_depths,
    #                              partial_basin_surface_depths, in_basin, extended_qualities_vector, logger,
    #                              gen_extract_velo_mod_call.topo_type)
    #
    #             for i in range(partial_global_mesh.nZ):
    #                 mid_pt_count = i * (1 + 2 * n_pts_smooth) + 1
    #                 mid_pt_count_plus = mid_pt_count + 1
    #                 mid_pt_count_minus = mid_pt_count - 1
    #
    #                 A = one_third * extended_qualities_vector.Rho[mid_pt_count_minus]
    #                 B = four_thirds * extended_qualities_vector.Rho[mid_pt_count]
    #                 C = one_third * extended_qualities_vector.Rho[mid_pt_count_plus]
    #                 partial_global_qualities.Rho[k][i] = half * (A + B + C)
    #
    #                 A = one_third * extended_qualities_vector.Vp[mid_pt_count_minus]
    #                 B = four_thirds * extended_qualities_vector.Vp[mid_pt_count]
    #                 C = one_third * extended_qualities_vector.Vp[mid_pt_count_plus]
    #                 partial_global_qualities.Vp[k][i] = half * (A + B + C)
    #
    #                 A = one_third * extended_qualities_vector.Vs[mid_pt_count_minus]
    #                 B = four_thirds * extended_qualities_vector.Vs[mid_pt_count]
    #                 C = one_third * extended_qualities_vector.Vs[mid_pt_count_plus]
    #                 partial_global_qualities.Vs[k][i] = half * (A + B + C)
    #         else:
    #             mesh_vector = extract_mesh_vector(partial_global_mesh, k)
    #             assign_qualities(global_model_parameters, velo_mod_1d_data, nz_tomography_data, global_surfaces,
    #                              basin_data, mesh_vector, partial_global_surface_depths, partial_basin_surface_depths,
    #                              in_basin, qualities_vector, calculation_log, gen_extract_velo_mod_call.topo_type)
    #
    #             for i in range(partial_global_mesh.nZ):
    #                 partial_global_qualities.Rho[k][i] = qualities_vector.Rho[i]
    #                 partial_global_qualities.Vp[k][i] = qualities_vector.Vp[i]
    #                 partial_global_qualities.Vs[k][i] = qualities_vector.Vs[i]
    #                 partial_global_qualities.inbasin[k][i] = qualities_vector.inbasin[i]
    #
    #     write_global_qualities(output_dir, partial_global_mesh, partial_global_qualities, gen_extract_velo_mod_call,
    #                            calculation_log, j)



    # def process_j(j):
    #     # do something with j
    #
    #     return j
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_j, j) for j in range(global_mesh.nY)]
    #     for future in concurrent.futures.as_completed(futures):
    #         j = future.result()
    #         print(f"\rGenerating velocity model {j * 100 / global_mesh.nY:.2f}% complete.", end="")
    #         sys.stdout.flush()



    print("\rGeneration of velocity model 100% complete.")
    print("Model generation complete.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate velocity model')
    parser.add_argument('vm_params', type=Path, help='Path to the vm_params.yaml file')
    parser.add_argument('out_dir', type=Path, help='Path to the output directory')
    parser.add_argument('--nzvm_registry', type=Path, help='Path to the nzvm_registry.yaml file', default=nzvm_registry_path)
    return parser.parse_args()

if __name__ == '__main__':

    global nzvm_registry

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    vm_params_path = args.vm_params
    assert vm_params_path.exists(), f"File is not present: {vm_params_path}"
    assert args.nzvm_registry.exists(), f"File is not present: {args.nzvm_registry}"

    out_dir= args.out_dir.resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f'Using vm_params file: {vm_params_path}')
    with open(vm_params_path, 'r') as f:
        vm_params = yaml.safe_load(f)


    with open(args.nzvm_registry) as f:
        nzvm_registry = yaml.safe_load(f)

    generate_velocity_model(out_dir, vm_params, logger)


