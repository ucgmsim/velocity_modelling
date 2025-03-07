"""
Velocity Model Registry Module.

This module provides the CVMRegistry class for managing and accessing velocity model data.
It serves as a central component for loading and retrieving various types of model data:
- 1D velocity models
- Basin models and surfaces
- Tomography data
- Global surface definitions

The registry loads model configuration from YAML files and provides methods to access
specific data components, handling file loading, caching, and path resolution.
"""

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

from velocity_modelling.cvm.constants import (
    DATA_ROOT,
    DEFAULT_OFFSHORE_1D_MODEL,
    DEFAULT_OFFSHORE_DISTANCE,
    NZVM_REGISTRY_PATH,
    VTYPE,
)
from velocity_modelling.cvm.geometry import (  # noqa: F401
    AdjacentPoints,
    MeshVector,
)
from velocity_modelling.cvm.logging import VMLogger


class CVMRegistry:
    """
    Registry for velocity model components.

    Manages loading, caching, and access to various velocity model components,
    including 1D models, basin data, surfaces, and tomography information.

    Attributes
    ----------
    registry : dict
        Loaded registry data from YAML configuration.
    version : str
        Version of the velocity model.
    vm_global_params : dict
        Global parameters for the specified model version.
    logger : VMLogger
        Logger for logging information and errors.
    cache : dict
        Cache for storing loaded data to prevent redundant file operations.
    """

    def __init__(
        self,
        version: str,
        logger: VMLogger = None,
        registry_path: Path = NZVM_REGISTRY_PATH,
    ):
        """
        Initialize the CVMRegistry.

        Parameters
        ----------
        version : str
            The version of the velocity model.
        logger : VMLogger, optional
            Logger for logging information and errors.
        registry_path : Path, optional
            The path to the registry file.
        """

        # Initialize VMLogger if not provided
        if logger is None:
            self.logger = VMLogger(name="velocity_model.registry")
        else:
            self.logger = logger

        with open(registry_path, "r") as f:
            self.registry = yaml.safe_load(f)
        self.version = version
        self.vm_global_params = None

        for vminfo in self.registry["vm"]:
            if str(vminfo["version"]) == version:
                self.vm_global_params = vminfo
                break

        if self.vm_global_params is None:
            raise ValueError(f"Version {version} not found in registry")

        self.logger = logger
        self.cache = {}  # Initialize a cache dictionary

    def log(self, message: str, level: int = VMLogger.INFO) -> None:
        """
        Log a message with the specified level.

        Parameters
        ----------
        message : str
            The message to log.
        level : int, optional
            The logging level (default is VMLogger.INFO).
        """

        if level is None:
            level = self.logger.INFO
        self.logger.log(message, level)

    def get_info(self, datatype: str, name: str) -> Optional[dict]:
        """
        Get information from the registry.

        Parameters
        ----------
        datatype : str
            The type of data to retrieve (e.g., 'tomography', 'surface', 'basin').
        name : str
            The name of the data entry.

        Returns
        -------
        dict or None
            The information dictionary for the specified data entry, or None if not found.
        """
        try:
            data_section = self.registry[datatype]
        except KeyError:
            self.log(f"Error: {datatype} not found in registry", VMLogger.ERROR)
            return None

        for info in data_section:
            assert (
                "name" in info
            ), f"Error: This entry in {datatype} has no name defined."
            if info["name"] == name:
                return info
        self.log(
            f"Error: {name} for datatype {datatype} not found in registry",
            VMLogger.ERROR,
        )
        return None

    def get_full_path(self, relative_path: Union[Path, str]) -> Path:
        """
        Get the full path for a given relative path.

        Parameters
        ----------
        relative_path : Path or str
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

    def load_1d_velo_sub_model(self, v1d_path: Path):
        """
        Load a 1D velocity submodel into memory.

        Parameters
        ----------
        v1d_path : Path or str
            The path to the 1D velocity model file.

        Returns
        -------
        VelocityModel1D
            The loaded 1D velocity model data.

        Raises
        ------
        SystemExit
            If the file cannot be found or read.
        """
        from velocity_modelling.cvm.velocity1d import (
            VelocityModel1D,
        )

        v1d_path = self.get_full_path(v1d_path)

        # Check if the data is already in the cache
        if v1d_path in self.cache:
            self.log(f"{v1d_path} loaded from CACHE", VMLogger.INFO)
            return self.cache[v1d_path]

        try:
            with open(v1d_path, "r") as file:
                next(file)
                data = np.loadtxt(file)
                velo_mod_1d_data = VelocityModel1D(
                    data[:, 0], data[:, 1], data[:, 2], data[:, 5]
                )
                # Store the loaded data in the cache
                self.cache[v1d_path] = velo_mod_1d_data
                self.log(f"Loaded 1D velocity model from {v1d_path}", VMLogger.INFO)

        except FileNotFoundError:
            self.log(
                f"Error: 1D velocity model file {v1d_path} not found.", VMLogger.ERROR
            )
            sys.exit(1)
        except Exception as e:
            self.log(f"Error loading 1D velocity model: {e}", VMLogger.ERROR)
            sys.exit(1)

        return velo_mod_1d_data

    def load_basin_data(self, basin_names: list[str]):
        """
        Load all basin data into the basin_data structure.

        Parameters
        ----------
        basin_names : list[str]
            List of basin names to load.

        Returns
        -------
        list[BasinData]
            List of loaded basin data.
        """
        from velocity_modelling.cvm.basin_model import (
            BasinData,
        )

        all_basin_data = []
        for basin_name in basin_names:
            self.log(f"Loading basin data for {basin_name}", VMLogger.INFO)
            basin_data = BasinData(self, basin_name, self.logger)
            all_basin_data.append(basin_data)
        return all_basin_data

    def load_basin_boundary(self, basin_boundary_path: Union[Path, str]):
        """
        Load a basin boundary from a file.

        Parameters
        ----------
        basin_boundary_path : Path or str
            The path to the basin boundary file.

        Returns
        -------
        np.ndarray
            The loaded basin boundary data as a Nx2 array of [longitude, latitude] points.

        Raises
        ------
        SystemExit
            If the file cannot be found or read, or if the boundary is not closed.
        """
        try:
            basin_boundary_path = self.get_full_path(basin_boundary_path)
            data = np.loadtxt(basin_boundary_path)
            lon = data[:, 0]
            lat = data[:, 1]
            boundary_data = np.column_stack((lon, lat))

            # Check if the boundary is closed (first and last points match)
            if lon[-1] != lon[0] or lat[-1] != lat[0]:
                self.log(
                    f"Error: Basin boundary {basin_boundary_path} is not closed.",
                    VMLogger.ERROR,
                )
                sys.exit(1)

            return boundary_data

        except FileNotFoundError:
            self.log(
                f"Error: Basin boundary file {basin_boundary_path} not found.",
                VMLogger.ERROR,
            )
            sys.exit(1)
        except Exception as e:
            self.log(
                f"Error reading basin boundary file {basin_boundary_path}: {e}",
                VMLogger.ERROR,
            )
            sys.exit(1)

    def load_basin_submodel(self, basin_surface: dict):
        """
        Load a basin sub-model into the basin_data structure.

        Parameters
        ----------
        basin_surface : dict
            Dictionary containing basin surface data with keys 'name' and 'submodel'.

        Returns
        -------
        tuple or None
            A tuple of (submodel_name, submodel_data) or None if not applicable.

        Raises
        ------
        SystemExit
            If the specified submodel cannot be found.
        """
        submodel_name = basin_surface["submodel"]
        if submodel_name == "null":
            return None
        submodel = self.get_info("submodel", submodel_name)

        if submodel is None:
            self.log(f"Error: Submodel {submodel_name} not found.", VMLogger.ERROR)
            sys.exit(1)

        if submodel["type"] == "vm1d":
            vm1d = self.get_info("vm1d", submodel["name"])
            if vm1d is None:
                self.log(f"Error: vm1d {submodel['name']} not found.", VMLogger.ERROR)
                sys.exit(1)
            return (submodel_name, self.load_1d_velo_sub_model(vm1d["path"]))

        # TODO: investigate why the original code ignores non-vm1d submodels
        elif submodel["type"] == "relation":
            self.log(
                f"Using relation submodel {submodel_name} with no additional data",
                VMLogger.DEBUG,
            )
            return (submodel_name, None)
        elif submodel["type"] == "perturbation":
            self.log(
                f"Using perturbation submodel {submodel_name} with no additional data",
                VMLogger.DEBUG,
            )
            return (submodel_name, None)

    def load_basin_surface(self, basin_surface: dict):
        """
        Load a basin surface from a file.

        Parameters
        ----------
        basin_surface : dict
            Dictionary with keys 'name' and 'submodel' specifying the basin surface.

        Returns
        -------
        BasinSurfaceRead
            The loaded basin surface data.

        Raises
        ------
        SystemExit
            If the surface file cannot be found or read.
        """
        from velocity_modelling.cvm.basin_model import (
            BasinSurfaceRead,
        )

        surface_info = self.get_info("surface", basin_surface["name"])

        if surface_info is None:
            self.log(
                f"Error: Surface {basin_surface['name']} not found.", VMLogger.ERROR
            )
            sys.exit(1)

        self.log(f"Loading basin surface file {surface_info['path']}", VMLogger.DEBUG)

        basin_surface_path = self.get_full_path(surface_info["path"])
        # Check if the data is already in the cache
        if basin_surface_path in self.cache:
            self.log(f"{basin_surface_path} loaded from cache", VMLogger.DEBUG)
            return self.cache[basin_surface_path]

        try:
            with open(basin_surface_path, "r") as f:
                nlat, nlon = map(int, f.readline().split())
                basin_surf_read = BasinSurfaceRead(nlat, nlon)

                latitudes = np.fromfile(f, dtype=float, count=nlat, sep=" ")
                longitudes = np.fromfile(f, dtype=float, count=nlon, sep=" ")

                basin_surf_read.lats = latitudes
                basin_surf_read.lons = longitudes

                raster_data = np.fromfile(f, dtype=float, count=nlat * nlon, sep=" ")
                if len(raster_data) != nlat * nlon:
                    self.log(
                        f"Warning: In {basin_surface_path} raster data length mismatch: "
                        f"{len(raster_data)} != {nlat * nlon}. Padding with zeros.",
                        VMLogger.WARNING,
                    )
                    raster_data = np.pad(
                        raster_data, (0, nlat * nlon - len(raster_data)), "constant"
                    )

                basin_surf_read.raster = raster_data.reshape((nlat, nlon)).T

                # Calculate min/max extents
                first_lat = basin_surf_read.lats[0]
                last_lat = basin_surf_read.lats[nlat - 1]
                basin_surf_read.max_lat = np.maximum(first_lat, last_lat)
                basin_surf_read.min_lat = np.minimum(first_lat, last_lat)

                first_lon = basin_surf_read.lons[0]
                last_lon = basin_surf_read.lons[nlon - 1]
                basin_surf_read.max_lon = np.maximum(first_lon, last_lon)
                basin_surf_read.min_lon = np.minimum(first_lon, last_lon)

                # Store the loaded data in the cache
                self.cache[basin_surface_path] = basin_surf_read

                return basin_surf_read

        except FileNotFoundError:
            self.log(
                f"Error: Basin surface file {basin_surface_path} not found.",
                VMLogger.ERROR,
            )
            sys.exit(1)
        except Exception as e:
            self.log(f"Error loading basin surface: {e}", VMLogger.ERROR)
            sys.exit(1)

    def load_tomo_surface_data(
        self,
        tomo_name: str,
        offshore_surface_name: str = DEFAULT_OFFSHORE_DISTANCE,
        offshore_v1d_name: str = DEFAULT_OFFSHORE_1D_MODEL,
    ):
        """
        Load tomography surface data from registry.

        Reads tomography data including surface depths, VS30 values, and velocity surfaces
        for multiple depths and velocity types (Vp, Vs, density).

        Parameters
        ----------
        tomo_name : str
            The name of the tomography data in the registry.
        offshore_surface_name : str, optional
            The name of the offshore distance surface in the registry.
        offshore_v1d_name : str, optional
            The name of the offshore 1D velocity model in the registry.

        Returns
        -------
        TomographyData
            Object containing tomography surfaces and related metadata.

        Raises
        ------
        SystemExit
            If required files cannot be found or read.
        AssertionError
            If tomography surface files don't exist.
        """
        from velocity_modelling.cvm.global_model import (
            TomographyData,
        )

        tomo = self.get_info("tomography", tomo_name)
        surf_depth = tomo["elev"]
        special_offshore_tapering = tomo["special_offshore_tapering"]
        vs30 = self.load_global_surface(tomo["vs30_path"])

        self.log(
            f"Loading tomography data '{tomo_name}' with {len(surf_depth)} depth levels",
            VMLogger.INFO,
        )

        surfaces = []
        for i, elev in enumerate(surf_depth):
            surfaces.append({})
            elev_name = (
                f"{elev}" if elev == int(elev) else f"{elev:.2f}".replace(".", "p")
            )
            for vtype in VTYPE:
                tomofile = (
                    self.get_full_path(tomo["path"])
                    / f"surf_tomography_{vtype.name}_elev{elev_name}.in"
                )
                if not tomofile.exists():
                    self.log(
                        f"Error: Tomography file {tomofile} not found", VMLogger.ERROR
                    )
                    assert (
                        tomofile.exists()
                    ), f"Tomography file {tomofile} does not exist"

                surfaces[i][vtype.name] = self.load_global_surface(tomofile)
                self.log(
                    f"Loaded tomography surface for {vtype.name} at elevation {elev}",
                    VMLogger.DEBUG,
                )

        offshore_distance_surface = self.load_global_surface(
            self.get_info("surface", offshore_surface_name)["path"]
        )
        offshore_basin_model_1d = self.load_1d_velo_sub_model(
            self.get_info("vm1d", offshore_v1d_name)["path"]
        )

        tomography_data = TomographyData(
            name=tomo_name,
            surf_depth=surf_depth,
            special_offshore_tapering=special_offshore_tapering,
            vs30=vs30,
            surfaces=surfaces,
            offshore_distance_surface=offshore_distance_surface,
            offshore_basin_model_1d=offshore_basin_model_1d,
        )

        return tomography_data

    def load_all_global_data(self):
        """
        Load all data required for the 3D velocity model generation.

        This is the main data loader that assembles all components needed for the CVM:
        - 1D reference velocity model
        - Tomography data
        - Global surfaces (like topography, basin interfaces)
        - Basin models and boundaries


        Returns
        -------
        Tuple[VelocityModel1D, TomographyData, GlobalSurfaces, list[BasinData]]
            Tuple containing:
            - 1D velocity model data
            - Tomography data with surfaces
            - Global surfaces (topography, etc.)
            - List of basin data objects
        """
        velo_mod_1d_data = None
        nz_tomography_data = None

        global_model_params = self.vm_global_params

        self.log("Loading global velocity submodel data", VMLogger.INFO)
        for submodel in global_model_params["submodels"]:
            if submodel == "v1DsubMod":
                velo_mod_1d_data = self.load_1d_velo_sub_model(submodel)
                self.log("Loaded 1D velocity model data", VMLogger.INFO)
            elif submodel == "NaNsubMod":
                self.log("NaN submodel recognized (no data to load)", VMLogger.DEBUG)
            else:
                nz_tomography_data = self.load_tomo_surface_data(
                    global_model_params["tomography"]
                )
                self.log(
                    f"Loaded tomography data: {global_model_params['tomography']}",
                    VMLogger.INFO,
                )

        if nz_tomography_data is not None:
            self.log(
                "Loading smooth boundaries for tomography transitions", VMLogger.INFO
            )
            nz_tomography_data.smooth_boundary = self.load_smooth_boundaries(
                global_model_params["basins"]
            )

        self.log("Loading global surfaces", VMLogger.INFO)
        global_surfaces = self.load_global_surface_data(global_model_params["surfaces"])

        self.log("Loading basin data", VMLogger.INFO)
        basin_data = self.load_basin_data(global_model_params["basins"])

        self.log("All global data successfully loaded", VMLogger.INFO)
        return velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data

    def load_global_surface(self, surface_file: Union[Path, str]):
        """
        Load a global surface raster from a file.

        Reads a surface defined by a grid of latitude, longitude points and their
        corresponding values (e.g., elevation, velocity).

        Parameters
        ----------
        surface_file : Path or str
            Path to the global surface file.

        Returns
        -------
        GlobalSurfaceRead
            Object containing the surface grid data with latitude, longitude, and values.

        Raises
        ------
        SystemExit
            If the file cannot be found or read properly.
        """

        from velocity_modelling.cvm.global_model import (
            GlobalSurfaceRead,
        )

        surface_file = self.get_full_path(surface_file)

        try:
            with open(surface_file, "r") as f:
                nlat, nlon = map(int, f.readline().split())
                self.log(
                    f"Reading surface with dimensions {nlat}x{nlon} from {surface_file}",
                    VMLogger.DEBUG,
                )

                latitudes = np.fromfile(f, dtype=float, count=nlat, sep=" ")
                longitudes = np.fromfile(f, dtype=float, count=nlon, sep=" ")

                raster_data = np.fromfile(f, dtype=float, count=nlat * nlon, sep=" ")
                if len(raster_data) != nlat * nlon:
                    error_msg = (
                        f"Data length mismatch in {surface_file}: "
                        f"got {len(raster_data)}, expected {nlat * nlon}. Missing data will be padded with 0."
                    )
                    self.log(error_msg, VMLogger.WARNING)
                    raster_data = np.pad(
                        raster_data, (0, nlat * nlon - len(raster_data)), "constant"
                    )

                return GlobalSurfaceRead(
                    latitudes, longitudes, raster_data.reshape((nlat, nlon)).T
                )

        except FileNotFoundError:
            self.log(f"Surface file {surface_file} not found", VMLogger.ERROR)
            sys.exit(1)
        except Exception as e:
            self.log(f"Error reading surface file {surface_file}: {e}", VMLogger.ERROR)
            sys.exit(1)

    def load_global_surface_data(self, global_surface_names: list[str]):
        """
        Load multiple global surfaces from the registry.

        Parameters
        ----------
        global_surface_names : list[str]
            List of global surface names defined in the registry.

        Returns
        -------
        GlobalSurfaces
            Container object with all loaded global surfaces.
        """
        from velocity_modelling.cvm.global_model import (
            GlobalSurfaces,
        )

        surfaces = []

        for name in global_surface_names:
            surface_info = self.get_info("surface", name)
            if surface_info:
                self.log(f"Loading global surface: {name}", VMLogger.DEBUG)
                surfaces.append(self.load_global_surface(surface_info["path"]))
            else:
                self.log(
                    f"Warning: Surface {name} not found in registry", VMLogger.WARNING
                )

        global_surfaces = GlobalSurfaces(surfaces)
        self.log(f"Loaded {len(global_surface_names)} global surfaces", VMLogger.INFO)
        return global_surfaces

    def load_smooth_boundaries(self, basin_names: list[str]):
        """
        Load smoothing boundary data for model transitions.

        Creates a collection of boundary points that define where velocity models
        should be smoothly transitioned between basins and background models.

        Parameters
        ----------
        basin_names : list[str]
            Names of basins to load smoothing boundaries for.

        Returns
        -------
        SmoothingBoundary
            Object containing smoothing boundary points.
        """
        from velocity_modelling.cvm.geometry import (
            SmoothingBoundary,
        )

        smooth_bound_lons = []
        smooth_bound_lats = []
        count = 0

        self.log(
            f"Loading smoothing boundaries for {len(basin_names)} basins", VMLogger.INFO
        )

        for basin_name in basin_names:
            basin = self.get_info("basin", basin_name)
            if basin is None:
                self.log(f"Basin {basin_name} not found in registry", VMLogger.ERROR)
                sys.exit(1)

            if "smoothing" in basin:
                boundary_vec_filename = self.get_full_path(basin["smoothing"])

                if boundary_vec_filename.exists():
                    self.log(
                        f"Loading smoothing boundary: {boundary_vec_filename}",
                        VMLogger.INFO,
                    )
                    try:
                        data = np.fromfile(boundary_vec_filename, dtype=float, sep=" ")
                        smooth_lons = data[0::2]
                        smooth_lats = data[1::2]
                        smooth_bound_lons.extend(smooth_lons)
                        smooth_bound_lats.extend(smooth_lats)
                        count += len(smooth_lons)
                        self.log(
                            f"Added {len(smooth_lons)} smoothing points for basin {basin_name}",
                            VMLogger.DEBUG,
                        )
                    except Exception as e:
                        self.log(
                            f"Error reading smoothing boundary: {e}", VMLogger.ERROR
                        )
                else:
                    self.log(
                        f"Smoothing boundary file not found: {boundary_vec_filename} -- to be ignored",
                        VMLogger.WARNING,
                    )
            else:
                self.log(f"No smoothing defined for basin {basin_name}", VMLogger.DEBUG)

        self.log(f"Total smoothing boundary points: {count}", VMLogger.INFO)
        return SmoothingBoundary(smooth_bound_lons, smooth_bound_lats)
