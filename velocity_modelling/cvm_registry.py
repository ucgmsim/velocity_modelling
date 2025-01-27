import yaml

from enum import Enum
import numpy as np
from typing import List, Dict
from pathlib import Path

from logging import Logger
import logging
import sys

DATA_ROOT = Path(__file__).parent / "Data"
nzvm_registry_path = DATA_ROOT / "nzvm_registry.yaml"

DEFAULT_OFFSHORE_1D_MODEL = "Cant1D_v2"  # vm1d name for offshore 1D model
DEFAULT_OFFSHORE_DISTANCE = "offshore"  # surface name for offshore distance


class CVMRegistry:  # Forward declaration
    pass


class TomographyData:
    def __init__(
        self,
        cvm_registry: CVMRegistry,
        tomo_name: str,
        offshore_surface_name: str,
        offshore_v1d_name: str,
        logger: Logger = None,
    ):
        """
        Initialize the TomographyData.

        Parameters
        ----------
        cvm_registry : CVMRegistry
            The CVMRegistry instance.
        tomo_name : str
            The name of the tomography data.
        offshore_surface_name : str
            The name of the offshore surface.
        offshore_v1d_name : str
            The name of the offshore 1D model.
        logger : Logger, optional

        """
        tomo = cvm_registry.get_info("tomography", tomo_name)
        self.name = tomo_name

        self.surfDeps = tomo["elev"]
        self.surf = []

        self.tomography_loaded = False
        self.special_offshore_tapering = tomo["special_offshore_tapering"]
        self.smooth_boundary = SmoothingBoundary()

        surf_tomo_path = cvm_registry.get_full_path(tomo["path"])
        offshore_surface_path = cvm_registry.get_info("surface", offshore_surface_name)[
            "path"
        ]
        offshore_v1d_path = cvm_registry.get_info("vm1d", offshore_v1d_name)["path"]

        self.vs30 = cvm_registry.load_global_surface(tomo["vs30_path"])

        for i in range(len(self.surfDeps)):
            self.surf.append({})  # self.surf[i] is an empty dictionary
            elev = self.surfDeps[i]
            if elev == int(elev):  # if the elevation is an integer
                elev_name = f"{elev}"
            else:
                elev_name = f"{elev:.2f}".replace(".", "p")

            for vtype in VTYPE:
                tomofile = (
                    surf_tomo_path / f"surf_tomography_{vtype.name}_elev{elev_name}.in"
                )
                self.surf[i][vtype.name] = cvm_registry.load_global_surface(tomofile)

        self.offshore_distance_surface = cvm_registry.load_global_surface(
            offshore_surface_path
        )
        self.offshore_basin_model_1d = cvm_registry.load_1d_velo_sub_model(
            offshore_v1d_path
        )
        self.tomography_loaded = True
        self.logger = logger

    def log(self, message, level=logging.INFO):
        if self.logger is not None:
            self.logger.log(level, message)
        else:
            print(message, file=sys.stderr)


def check_boundary_index(func):
    def wrapper(self, i, *args, **kwargs):
        if i < 0 or i >= len(self.boundary):
            self.log(
                f"Error: basin boundary {i} not found. Max index is {len(self.boundary) - 1}"
            )
            return None
        return func(self, i, *args, **kwargs)

    return wrapper


class BasinData:
    def __init__(
        self, cvm_registry: CVMRegistry, basin_name: str, logger: Logger = None
    ):
        """
        Initialize the BasinData.

        Parameters
        ----------
        cvm_registry : CVMRegistry
            The CVMRegistry instance.
        basin_name : str
            The name of the basin.
        logger : Logger, optional
            The logger instance.
        """
        self.name = basin_name
        basin_info = cvm_registry.get_info("basin", basin_name)

        self.surf = [
            cvm_registry.load_basin_surface(surface["path"])
            for surface in basin_info["surfaces"]
        ]
        self.boundary = [
            cvm_registry.load_basin_boundary(boundary["path"])
            for boundary in basin_info["boundaries"]
        ]
        self.submodel = [
            cvm_registry.load_basin_submodel(surface)
            for surface in basin_info["surfaces"]
        ]
        self.perturbation_data = None
        self.logger = logger

        self.log(f"Basin {basin_name} fully loaded.")

    def log(self, message, level=logging.INFO):
        if self.logger is not None:
            self.logger.log(level, message)
        else:
            print(message, file=sys.stderr)

    @check_boundary_index
    def boundary_lat(self, i: int) -> np.ndarray:
        """
        Get the latitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        np.ndarray
            The latitude of the boundary.
        """
        return self.boundary[i][:, 1]

    @check_boundary_index
    def boundary_lon(self, i: int) -> np.ndarray:
        """
        Get the longitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        np.ndarray
            The longitude of the boundary.
        """
        return self.boundary[i][:, 0]

    @check_boundary_index
    def min_lon_boundary(self, i: int) -> float:
        """
        Get the minimum longitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        float
            The minimum longitude of the boundary.
        """
        return np.min(self.boundary_lon(i))

    @check_boundary_index
    def max_lon_boundary(self, i: int) -> float:
        """
        Get the maximum longitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        float
            The maximum longitude of the boundary.
        """
        return np.max(self.boundary_lon(i))

    @check_boundary_index
    def min_lat_boundary(self, i: int) -> float:
        """
        Get the minimum latitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        float
            The minimum latitude of the boundary.
        """
        return np.min(self.boundary_lat(i))

    @check_boundary_index
    def max_lat_boundary(self, i: int) -> float:
        """
        Get the maximum latitude of the boundary at index i.

        Parameters
        ----------
        i : int
            The index of the boundary.

        Returns
        -------
        float
            The maximum latitude of the boundary.
        """
        return np.max(self.boundary_lat(i))


class BasinSurfaceRead:
    def __init__(self, nLat: int, nLon: int):
        """
        Initialize the BasinSurfaceRead.

        Parameters
        ----------
        nLat : int
            The number of latitude points.
        nLon : int
            The number of longitude points.
        """
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
    def __init__(self, nX: int, nY: int, nZ: int):
        """
        Initialize the GlobalMesh.

        Parameters
        ----------
        nX : int
            The number of points in the X direction.
        nY : int
            The number of points in the Y direction.
        nZ : int
            The number of points in the Z direction.
        """
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
    def __init__(self, nX: int, nZ: int):
        """
        Initialize the PartialGlobalMesh.

        Parameters
        ----------
        nX : int
            The number of points in the X direction.
        nZ : int
            The number of points in the Z direction.
        """
        self.Lon = np.zeros(nX)
        self.Lat = np.zeros(nX)
        self.X = np.zeros(nX)
        self.Z = np.zeros(nZ)
        self.nX = nX
        self.nY = 1
        self.nZ = nZ
        self.Y = 0.0


class GlobalSurfaces:
    def __init__(self):
        """
        Initialize the GlobalSurfaces.
        """
        self.surf = []


class GlobalSurfaceRead:
    def __init__(self, nLat: int, nLon: int):
        """
        Initialize the GlobalSurfaceRead.

        Parameters
        ----------
        nLat : int
            The number of latitude points.
        nLon : int
            The number of longitude points.
        """
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
        """
        Initialize the ModelExtent.

        Parameters
        ----------
        vm_params : Dict
            The velocity model parameters.
        """
        self.originLat = vm_params["MODEL_LAT"]
        self.originLon = vm_params["MODEL_LON"]
        self.originRot = vm_params["MODEL_ROT"]  # in degrees
        self.Xmax = vm_params["extent_x"]
        self.Ymax = vm_params["extent_y"]
        self.Zmax = vm_params["extent_zmax"]
        self.Zmin = vm_params["extent_zmin"]
        self.hDep = vm_params["hh"]
        self.hLatLon = vm_params["hh"]
        self.nx = vm_params["nx"]
        self.ny = vm_params["ny"]
        self.nz = vm_params["nz"]


class SmoothingBoundary:
    def __init__(self):
        """
        Initialize the SmoothingBoundary.
        """
        self.n = 0
        self.xPts = []
        self.yPts = []


class VeloMod1DData:
    def __init__(self):
        """
        Initialize the VeloMod1DData.
        """
        self.Vp = []
        self.Vs = []
        self.Rho = []
        self.Dep = []
        self.nDep = 0


class VTYPE(Enum):
    vp = 0
    vs = 1
    rho = 2


class CVMRegistry:
    def __init__(
        self,
        version: str,
        registry_path: Path = nzvm_registry_path,
        logger: Logger = None,
    ):
        """
        Initialize the CVMRegistry.

        Parameters
        ----------
        version : str
            The version of the velocity model.
        registry_path : Path, optional
            The path to the registry file (default is nzvm_registry_path).
        """
        with open(nzvm_registry_path, "r") as f:
            self.registry = yaml.safe_load(f)
        self.version = version
        self.vm_global_params = None

        for vminfo in self.registry["vm"]:
            if str(vminfo["version"]) == version:
                self.vm_global_params = vminfo
                break

        self.logger = logger

    def log(self, message, level=logging.INFO):
        if self.logger is not None:
            self.logger.log(level, message)
        else:
            print(message, file=sys.stderr)

    def get_info(self, datatype: str, name: str) -> Dict:
        """
        Get information from the registry.

        Parameters
        ----------
        datatype : str
            The type of data to retrieve (e.g., 'tomography', 'surface').
        name : str
            The name of the data entry.

        Returns
        -------
        Dict
            The information dictionary for the specified data entry.
        """
        try:
            self.registry[datatype]
        except KeyError:
            self.log(f"Error: {datatype} not found in registry")
            return None

        for info in self.registry[datatype]:
            assert (
                "name" in info
            ), f"Error: This entry in {datatype} has no name defined."
            if info["name"] == name:
                return info
        self.log(f"Error: {name} for datatype {datatype} not found in registry")
        return None

    def get_full_path(self, relative_path: Path) -> Path:
        """
        Get the full path for a given relative path.

        Parameters
        ----------
        relative_path : Path
            The relative path to convert.

        Returns
        -------
        Path
            The full path.
        """
        return (
            DATA_ROOT / relative_path
            if not Path(relative_path).is_absolute()
            else Path(relative_path)
        )

    def load_1d_velo_sub_model(self, v1d_path: Path) -> VeloMod1DData:
        """
        Load a 1D velocity submodel into memory.

        Parameters
        ----------
        v1d_path : Path
            The path to the 1D velocity model file.

        Returns
        -------
        VeloMod1DData
            The loaded 1D velocity model data.
        """
        v1d_path = self.get_full_path(v1d_path)
        velo_mod_1d_data = VeloMod1DData()
        try:
            with open(v1d_path, "r") as file:
                next(file)
                data = np.loadtxt(file)
                velo_mod_1d_data.Vp = data[:, 0].tolist()
                velo_mod_1d_data.Vs = data[:, 1].tolist()
                velo_mod_1d_data.Rho = data[:, 2].tolist()
                velo_mod_1d_data.Dep = data[:, 5].tolist()
                velo_mod_1d_data.nDep = len(velo_mod_1d_data.Dep)
        except FileNotFoundError:
            self.log(f"Error 1D velocity model file {v1d_path} not found.")
            exit(1)

        return velo_mod_1d_data

    def load_basin_data(self, basin_names: List[str]):
        """
        Load all basin data into the basin_data structure.

        Parameters
        ----------
        basin_names : List[str]
            List of basin names to load.

        Returns
        -------
        List[BasinData]
            List of loaded basin data.
        """
        all_basin_data = []
        for basin_name in basin_names:
            basin_data = BasinData(self, basin_name)
            all_basin_data.append(basin_data)
        return all_basin_data

    def load_basin_boundary(self, basin_boundary_path: Path):
        """
        Load a basin boundary from a file.

        Parameters
        ----------
        basin_boundary_path : Path
            The path to the basin boundary file.

        Returns
        -------
        np.ndarray
            The loaded basin boundary data.
        """
        try:
            basin_boundary_path = self.get_full_path(basin_boundary_path)
            data = np.loadtxt(basin_boundary_path)
            lon = data[:, 0]
            lat = data[:, 1]
            boundary_data = np.column_stack((lon, lat))

            assert lon[-1] == lon[0]
            assert lat[-1] == lat[0]
        except FileNotFoundError:
            self.log(f"Error basin boundary file {basin_boundary_path} not found.")
            exit(1)
        except Exception as e:
            self.log(f"Error reading basin boundary file {basin_boundary_path}: {e}")
            exit(1)
        return boundary_data

    def load_basin_submodel(self, basin_surface: dict):
        """
        Load a basin sub-model into the basin_data structure.

        Parameters
        ----------
        basin_surface : dict
            Dictionary containing basin surface data.

        Returns
        -------
        VeloMod1DData or None
            The loaded sub-model data or None if not applicable.
        """
        submodel_name = basin_surface["submodel"]
        if submodel_name == "null":
            return None
        submodel = self.get_info("submodel", submodel_name)

        if submodel is None:
            self.log(f"Error: submodel {submodel_name} not found.")
            exit(1)

        if submodel["type"] == "vm1d":
            vm1d = self.get_info("vm1d", submodel["name"])
            if vm1d is None:
                self.log(f"Error: vm1d {submodel['name']} not found.")
                exit(1)
            return self.load_1d_velo_sub_model(vm1d["path"])

        elif submodel["type"] == "relation":
            return VeloMod1DData()
        elif submodel["type"] == "perturbation":
            return VeloMod1DData()

    def load_basin_surface(self, basin_surface_path: Path):
        """
        Load a basin surface from a file.

        Parameters
        ----------
        basin_surface_path : Path
            The path to the basin surface file.

        Returns
        -------
        BasinSurfaceRead
            The loaded basin surface data.
        """
        self.log(f"Loading basin surface file {basin_surface_path}")

        basin_surface_path = self.get_full_path(basin_surface_path)

        try:
            with open(basin_surface_path, "r") as f:
                nLat, nLon = map(int, f.readline().split())
                basin_surf_read = BasinSurfaceRead(nLat, nLon)

                latitudes = np.fromfile(f, dtype=float, count=nLat, sep=" ")
                longitudes = np.fromfile(f, dtype=float, count=nLon, sep=" ")

                basin_surf_read.lati = latitudes
                basin_surf_read.loni = longitudes

                raster_data = np.fromfile(f, dtype=float, count=nLat * nLon, sep=" ")
                assert (
                    len(raster_data) == nLat * nLon
                ), f"Error: in {basin_surface_path} raster data length mismatch: {len(raster_data)} != {nLat * nLon}"
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
            self.log(f"Error basin surface file {basin_surface_path} not found.")
            exit(1)
        except Exception as e:
            self.log(f"Error: {e}")
            exit(1)

    def load_tomo_surface_data(
        self,
        tomo_name: str,
        offshore_surface_name: str = DEFAULT_OFFSHORE_DISTANCE,
        offshore_v1d_name: str = DEFAULT_OFFSHORE_1D_MODEL,
    ) -> TomographyData:
        """
        Load tomography surface data.

        Parameters
        ----------
        tomo_name : str
            The name of the tomography data.
        offshore_surface_name : str, optional
            The name of the offshore surface (default is DEFAULT_OFFSHORE_DISTANCE).
        offshore_v1d_name : str, optional
            The name of the offshore 1D model (default is DEFAULT_OFFSHORE_1D_MODEL).

        Returns
        -------
        TomographyData
            The loaded tomography data.
        """
        return TomographyData(self, tomo_name, offshore_surface_name, offshore_v1d_name)

    def load_all_global_data(self, logger: Logger):
        """
        Load all data required to generate the velocity model, global surfaces, sub velocity models, and all basin data.

        Parameters
        ----------
        logger : Logger
            Logger for logging information.

        Returns
        -------
        Tuple[VeloMod1DData, TomographyData, GlobalSurfaces, List[BasinData]]
            The loaded global data.
        """
        velo_mod_1d_data = None
        nz_tomography_data = None

        global_model_params = self.vm_global_params

        self.log("Loading global velocity submodel data.")
        for i in range(len(global_model_params["velo_submodels"])):
            if global_model_params["velo_submodels"][i] == "v1DsubMod":
                velo_mod_1d_data = self.load_1d_velo_sub_model(
                    global_model_params["velo_submodels"][i]
                )
                self.log("Loaded 1D velocity model data.")
            elif global_model_params["velo_submodels"][i] == "NaNsubMod":
                pass
            else:
                nz_tomography_data = self.load_tomo_surface_data(
                    global_model_params["tomography"]
                )
                self.log("Loaded tomography data.")

        if nz_tomography_data is not None:
            self.load_smooth_boundaries(
                nz_tomography_data, global_model_params["basins"]
            )

        self.log("Completed loading of global velocity submodel data.")

        global_surfaces = self.load_global_surface_data(global_model_params["surfaces"])
        self.log("Completed loading of global surfaces.")

        self.log("Loading basin data.")
        basin_data = self.load_basin_data(global_model_params["basins"])
        self.log("Completed loading basin data.")
        self.log("All global data loaded.")
        return velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data

    def load_global_surface(self, surface_file: Path):
        """
        Load a global surface from a file.

        Parameters
        ----------
        surface_file : Path
            The path to the global surface file.

        Returns
        -------
        GlobalSurfaceRead
            The loaded global surface data.
        """
        surface_file = self.get_full_path(surface_file)

        try:
            with open(surface_file, "r") as f:
                nLat, nLon = map(int, f.readline().split())
                global_surf_read = GlobalSurfaceRead(nLat, nLon)

                latitudes = np.fromfile(f, dtype=float, count=nLat, sep=" ")
                longitudes = np.fromfile(f, dtype=float, count=nLon, sep=" ")

                global_surf_read.lati = latitudes
                global_surf_read.loni = longitudes

                raster_data = np.fromfile(f, dtype=float, count=nLat * nLon, sep=" ")
                try:
                    global_surf_read.raster = raster_data.reshape((nLon, nLat)).T
                except:
                    self.log(
                        f"Error: in {surface_file} raster data length mismatch: {len(raster_data)} != {nLat * nLon}"
                    )
                    exit(1)

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
            self.log(f"Error surface file {surface_file} not found.")
            exit(1)
        except Exception as e:
            self.log(f"Error: {e}")
            exit(1)

    def load_global_surface_data(self, global_surface_names: List[str]):
        """
        Load all global surface data.

        Parameters
        ----------
        global_surface_names : List[str]
            List of global surface names to load.

        Returns
        -------
        GlobalSurfaces
            The loaded global surfaces.
        """
        surfaces = [self.get_info("surface", name) for name in global_surface_names]
        global_surfaces = GlobalSurfaces()

        for surface in surfaces:
            global_surfaces.surf.append(self.load_global_surface(surface["path"]))

        return global_surfaces

    def load_smooth_boundaries(
        self, nz_tomography_data: TomographyData, basin_names: List[str]
    ):
        """
        Load smooth boundaries for the tomography data.

        Parameters
        ----------
        nz_tomography_data : TomographyData
            The tomography data to load smooth boundaries for.
        basin_names : List[str]
            List of basin names to load smooth boundaries for.
        """
        smooth_bound = nz_tomography_data.smooth_boundary
        count = 0

        for basin_name in basin_names:
            basin = self.get_info("basin", basin_name)
            if "smoothing" in basin:
                boundary_vec_filename = self.get_full_path(basin["smoothing"]["path"])

                if boundary_vec_filename.exists():
                    self.log(
                        f"Loading offshore smoothing file: {boundary_vec_filename}."
                    )
                    try:
                        data = np.fromfile(boundary_vec_filename, dtype=float, sep=" ")
                        x_pts = data[0::2]
                        y_pts = data[1::2]
                        smooth_bound.xPts.extend(x_pts)
                        smooth_bound.yPts.extend(y_pts)
                        count += len(x_pts)
                    except Exception as e:
                        self.log(
                            f"Error reading smoothing boundary vector file {boundary_vec_filename}: {e}"
                        )
                else:
                    self.log(
                        f"Error smoothing boundary vector file {boundary_vec_filename} not found."
                    )
            else:
                self.log(f"Smoothing not required for basin {basin_name}.")
        smooth_bound.n = count
