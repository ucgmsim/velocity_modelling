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

from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

from velocity_modelling.cvm.constants import (
    DATA_ROOT,
    DEFAULT_OFFSHORE_1D_MODEL,
    DEFAULT_OFFSHORE_DISTANCE,
    MODEL_VERSIONS_ROOT,
    NZVM_REGISTRY_PATH,
    VelocityTypes,
)
from velocity_modelling.cvm.geometry import SmoothingBoundary
from velocity_modelling.cvm.global_model import GlobalSurfaceRead
from velocity_modelling.cvm.logging import VMLogger


class CVMRegistry:
    """
    Registry for velocity model components.

    Manages loading, caching, and access to various velocity model components,
    including 1D models, basin data, surfaces, and tomography information.

    Parameters
    ----------
    version : str
        The version of the velocity model.
    logger : VMLogger, optional
        Logger for logging information and errors.
    registry_path : Path, optional
        The path to the registry file.

    Attributes
    ----------
    registry : dict
        Loaded registry data from YAML configuration.
    version : str
        Version of the velocity model.
    global_params : dict
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
            Logger instance for logging; created if not provided.
        registry_path : Path, optional
            Path to the registry YAML file.

        Raises
        ------
        ValueError
            If the model version file is not found or cannot be loaded.
        """
        # Initialize VMLogger if not provided
        self.logger = (
            logger if logger is not None else VMLogger(name="velocity_model.registry")
        )

        with open(registry_path, "r") as f:
            self.registry = yaml.safe_load(f)

        # Normalize version (replace '.' with 'p' if present)
        self.version = version.replace(".", "p")
        self.global_params = None

        model_version_path = MODEL_VERSIONS_ROOT / f"{self.version}.yaml"
        if not model_version_path.exists():
            raise ValueError(
                f"Model version file for {self.version} not found at {model_version_path}"
            )

        with open(model_version_path, "r") as f:
            self.global_params = yaml.safe_load(f)

        if self.global_params is None:
            raise ValueError(f"Failed to load recipe for version {self.version}")

        # Validate global_params
        global_surfaces_list = self.global_params.get("surfaces", [])
        if not global_surfaces_list:
            global_surfaces_list = []

        # Adjust surfaces to include posInfSurf and negInfSurf
        first_surface_name = (
            global_surfaces_list[0]["name"] if global_surfaces_list else None
        )
        last_surface_name = (
            global_surfaces_list[-1]["name"] if global_surfaces_list else None
        )
        if first_surface_name != "posInfSurf" or last_surface_name != "negInfSurf":
            self.global_params["surfaces"] = (
                [{"name": "posInfSurf", "submodel": "nan_submod"}]
                + global_surfaces_list
                + [{"name": "negInfSurf", "submodel": None}]
            )

        # Separate surface names and submodels
        self.global_params["surface_names"] = [
            d["name"] for d in self.global_params["surfaces"]
        ]
        self.global_params["submodels"] = [
            d["submodel"] for d in self.global_params["surfaces"]
        ][:-1]

        self.cache = {}  # Initialize cache

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
        self.logger.log(message, level or VMLogger.INFO)

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

        Raises
        ------
        KeyError
            If the specified name is not found in the datatype section.
        """
        try:
            data_section = self.registry[datatype]
        except KeyError:
            self.log(f"Error: {datatype} not found in registry", VMLogger.ERROR)
            return None

        for info in data_section:
            assert "name" in info, f"Error: Entry in {datatype} lacks a 'name' field."
            if info["name"] == name:
                return info
        self.log(
            f"Error: {name} for datatype {datatype} not found in registry",
            VMLogger.ERROR,
        )
        raise KeyError(f"{name} not found in {datatype}")

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
            The resolved full path.
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
        FileNotFoundError
            If the specified file cannot be found.
        RuntimeError
            If the file cannot be read or parsed correctly.
        """
        from velocity_modelling.cvm.velocity1d import VelocityModel1D

        v1d_path = self.get_full_path(v1d_path)
        if v1d_path in self.cache:
            self.log(f"{v1d_path} loaded from CACHE", VMLogger.DEBUG)
            return self.cache[v1d_path]

        try:
            with open(v1d_path, "r") as file:
                next(file)  # Skip header
                data = np.loadtxt(file)
                velo_mod_1d_data = VelocityModel1D(
                    data[:, 0], data[:, 1], data[:, 2], data[:, 5]
                )
                self.cache[v1d_path] = velo_mod_1d_data
                self.log(f"Loaded 1D velocity model from {v1d_path}", VMLogger.INFO)
                return velo_mod_1d_data
        except FileNotFoundError:
            self.log(
                f"Error: 1D velocity model file {v1d_path} not found.", VMLogger.ERROR
            )
            raise FileNotFoundError(f"1D velocity model file {v1d_path} not found")
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise  # Re-raise critical exceptions
            self.log(f"Error loading 1D velocity model: {e}", VMLogger.ERROR)
            raise RuntimeError(
                f"Failed to load 1D velocity model from {v1d_path}: {str(e)}"
            )

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
        from velocity_modelling.cvm.basin_model import BasinData

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
        FileNotFoundError
            If the boundary file cannot be found.
        ValueError
            If the boundary is not closed (first and last points do not match).
        RuntimeError
            If the file cannot be read or parsed correctly.
        """
        basin_boundary_path = self.get_full_path(basin_boundary_path)
        try:
            data = np.loadtxt(basin_boundary_path)
            lon, lat = data[:, 0], data[:, 1]
            boundary_data = np.column_stack((lon, lat))

            if lon[-1] != lon[0] or lat[-1] != lat[0]:
                self.log(
                    f"Error: Basin boundary {basin_boundary_path} is not closed.",
                    VMLogger.ERROR,
                )
                raise ValueError(f"Basin boundary {basin_boundary_path} is not closed")
            return boundary_data
        except FileNotFoundError:
            self.log(
                f"Error: Basin boundary file {basin_boundary_path} not found.",
                VMLogger.ERROR,
            )
            raise FileNotFoundError(
                f"Basin boundary file {basin_boundary_path} not found"
            )
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
            self.log(
                f"Error reading basin boundary file {basin_boundary_path}: {e}",
                VMLogger.ERROR,
            )
            raise RuntimeError(
                f"Failed to load basin boundary from {basin_boundary_path}: {str(e)}"
            )

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
        KeyError
            If the submodel or its associated data cannot be found in the registry.
        """
        try:
            submodel_name = basin_surface["submodel"]
        except KeyError:
            return None  # Basement surface has no submodel

        submodel = self.get_info("submodel", submodel_name)
        if submodel is None:
            self.log(f"Error: Submodel {submodel_name} not found.", VMLogger.ERROR)
            raise KeyError(f"Submodel {submodel_name} not found")

        if submodel["type"] == "vm1d":
            vm1d = self.get_info("vm1d", submodel["name"])
            if vm1d is None:
                self.log(f"Error: vm1d {submodel['name']} not found.", VMLogger.ERROR)
                raise KeyError(f"vm1d {submodel['name']} not found")
            return (submodel_name, self.load_1d_velo_sub_model(vm1d["path"]))
        elif submodel["type"] in {"relation", "perturbation"}:
            self.log(
                f"Using {submodel['type']} submodel {submodel_name} with no additional data",
                VMLogger.DEBUG,
            )
            return (submodel_name, None)
        return None

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
        KeyError
            If the surface name is not found in the registry.
        FileNotFoundError
            If the surface file cannot be found.
        RuntimeError
            If the file cannot be read or parsed correctly.
        """
        from velocity_modelling.cvm.basin_model import BasinSurfaceRead

        surface_info = self.get_info("surface", basin_surface["name"])
        if surface_info is None:
            self.log(
                f"Error: Surface {basin_surface['name']} not found.", VMLogger.ERROR
            )
            raise KeyError(f"Surface {basin_surface['name']} not found")

        self.log(f"Loading basin surface file {surface_info['path']}", VMLogger.DEBUG)
        basin_surface_path = self.get_full_path(surface_info["path"])
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
                basin_surf_read.max_lat = max(
                    basin_surf_read.lats[0], basin_surf_read.lats[-1]
                )
                basin_surf_read.min_lat = min(
                    basin_surf_read.lats[0], basin_surf_read.lats[-1]
                )
                basin_surf_read.max_lon = max(
                    basin_surf_read.lons[0], basin_surf_read.lons[-1]
                )
                basin_surf_read.min_lon = min(
                    basin_surf_read.lons[0], basin_surf_read.lons[-1]
                )

                self.cache[basin_surface_path] = basin_surf_read
                return basin_surf_read
        except FileNotFoundError:
            self.log(
                f"Error: Basin surface file {basin_surface_path} not found.",
                VMLogger.ERROR,
            )
            raise FileNotFoundError(
                f"Basin surface file {basin_surface_path} not found"
            )
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
            self.log(f"Error loading basin surface: {e}", VMLogger.ERROR)
            raise RuntimeError(
                f"Failed to load basin surface from {basin_surface_path}: {str(e)}"
            )

    def load_tomo_surface_data(
        self,
        tomo_name: str,
        offshore_surface_name: str = DEFAULT_OFFSHORE_DISTANCE,
        offshore_v1d_name: str = DEFAULT_OFFSHORE_1D_MODEL,
    ):
        """
        Load tomography surface data from registry.

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
        KeyError
            If tomography, offshore surface, or offshore 1D model data is not found.
        FileNotFoundError
            If a required tomography file is missing.
        AssertionError
            If tomography surface files don't exist.
        """
        from velocity_modelling.cvm.global_model import TomographyData

        tomo = self.get_info("tomography", tomo_name)
        if tomo is None:
            raise KeyError(f"Tomography data {tomo_name} not found")

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
            for vtype in VelocityTypes:
                tomofile = (
                    self.get_full_path(tomo["path"])
                    / f"surf_tomography_{vtype.name}_elev{elev_name}.in"
                )
                if not tomofile.exists():
                    self.log(
                        f"Error: Tomography file {tomofile} not found", VMLogger.ERROR
                    )
                    raise FileNotFoundError(
                        f"Tomography file {tomofile} does not exist"
                    )
                surfaces[i][vtype.name] = self.load_global_surface(tomofile)
                self.log(
                    f"Loaded tomography surface for {vtype.name} at elevation {elev}",
                    VMLogger.DEBUG,
                )

        offshore_surface_info = self.get_info("surface", offshore_surface_name)
        if offshore_surface_info is None:
            self.log(
                f"Error: Offshore distance surface {offshore_surface_name} not found",
                VMLogger.ERROR,
            )
            raise KeyError(
                f"Offshore distance surface {offshore_surface_name} not found"
            )
        offshore_distance_surface = self.load_global_surface(
            offshore_surface_info["path"]
        )

        offshore_v1d_info = self.get_info("vm1d", offshore_v1d_name)
        if offshore_v1d_info is None:
            self.log(
                f"Error: Offshore 1D model {offshore_v1d_name} not found",
                VMLogger.ERROR,
            )
            raise KeyError(f"Offshore 1D model {offshore_v1d_name} not found")
        offshore_basin_model_1d = self.load_1d_velo_sub_model(offshore_v1d_info["path"])

        return TomographyData(
            name=tomo_name,
            surf_depth=surf_depth,
            special_offshore_tapering=special_offshore_tapering,
            vs30=vs30,
            surfaces=surfaces,
            offshore_distance_surface=offshore_distance_surface,
            offshore_basin_model_1d=offshore_basin_model_1d,
        )

    def load_all_global_data(self):
        """
        Load all data required for the 3D velocity model generation.

        Returns
        -------
        tuple[VelocityModel1D, TomographyData, list[GlobalSurfaceRead], list[BasinData]]
            Tuple containing:
            - 1D velocity model data
            - Tomography data with surfaces
            - List of global surfaces data (topography, etc.)
            - List of basin data objects

        Raises
        ------
        KeyError
            If required submodel or tomography data is not found.
        """
        velo_mod_1d_data = None
        nz_tomography_data = None

        self.log("Loading global surfaces", VMLogger.INFO)
        global_surfaces = self.load_global_surface_data(
            self.global_params["surface_names"]
        )

        self.log("Loading global velocity submodel data", VMLogger.INFO)
        for submodel_name in self.global_params["submodels"]:
            submodel_info = self.get_info("submodel", submodel_name)
            if submodel_info is None:
                self.log(f"Error: Submodel {submodel_name} not found", VMLogger.ERROR)
                raise KeyError(f"Submodel {submodel_name} not found")

            if submodel_info["type"] is None:
                self.log("nan submodel recognized (no data to load)", VMLogger.DEBUG)
            elif submodel_info["type"] == "vm1d":
                velo_mod_1d_data = self.load_1d_velo_sub_model(submodel_info["name"])
                self.log("Loaded 1D velocity model data", VMLogger.INFO)
            elif submodel_info["type"] == "tomography":
                if self.global_params.get("tomography"):
                    nz_tomography_data = self.load_tomo_surface_data(
                        self.global_params["tomography"]
                    )
                else:
                    self.log("Error: Tomography data not found", VMLogger.ERROR)
                    raise KeyError("Tomography data not found")
            elif submodel_info["type"] == "relation":
                self.log(
                    f"Using relation submodel {submodel_name} with no additional data",
                    VMLogger.DEBUG,
                )
            else:
                self.log(
                    f"Error: Unknown submodel type {submodel_info['type']} to be ignored.",
                    VMLogger.INFO,
                )

        if nz_tomography_data:
            self.log(
                "Loading smooth boundaries for tomography transitions", VMLogger.INFO
            )
            nz_tomography_data.smooth_boundary = self.load_smooth_boundaries(
                self.global_params["basins"]
            )

        self.log("Loading basin data", VMLogger.INFO)
        basin_data = self.load_basin_data(self.global_params["basins"])

        self.log("All global data successfully loaded", VMLogger.INFO)
        return velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data

    def load_global_surface(self, surface_file: Path | str) -> GlobalSurfaceRead:
        """
        Load a global surface raster from a file.

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
        FileNotFoundError
            If the surface file cannot be found.
        RuntimeError
            If the file cannot be read or parsed correctly.
        """
        from velocity_modelling.cvm.global_model import GlobalSurfaceRead

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
                    self.log(
                        f"Data length mismatch in {surface_file}: got {len(raster_data)}, expected {nlat * nlon}. "
                        "Missing data will be padded with 0.",
                        VMLogger.WARNING,
                    )
                    raster_data = np.pad(
                        raster_data, (0, nlat * nlon - len(raster_data)), "constant"
                    )

                return GlobalSurfaceRead(
                    latitudes, longitudes, raster_data.reshape((nlat, nlon)).T
                )
        except FileNotFoundError:
            self.log(f"Surface file {surface_file} not found", VMLogger.ERROR)
            raise FileNotFoundError(f"Surface file {surface_file} not found")
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
            self.log(f"Error reading surface file {surface_file}: {e}", VMLogger.ERROR)
            raise RuntimeError(
                f"Failed to load global surface from {surface_file}: {str(e)}"
            )

    def load_global_surface_data(
        self, global_surface_names: list[str]
    ) -> list[GlobalSurfaceRead]:
        """
        Load multiple global surfaces from the registry.

        Parameters
        ----------
        global_surface_names : list[str]
            List of global surface names defined in the registry.

        Returns
        -------
        list[GlobalSurfaceRead]
            List of GlobalSurfaceRead objects containing the loaded surface data.
        """
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
        self.log(f"Loaded {len(global_surface_names)} global surfaces", VMLogger.INFO)
        return surfaces

    def load_smooth_boundaries(self, basin_names: list[str]) -> SmoothingBoundary:
        """
        Load smoothing boundary data for model transitions.

        Parameters
        ----------
        basin_names : list[str]
            Names of basins to load smoothing boundaries for.

        Returns
        -------
        SmoothingBoundary
            Object containing smoothing boundary points.

        Raises
        ------
        KeyError
            If a basin is not found in the registry.
        RuntimeError
            If a smoothing boundary file cannot be read.
        """
        from velocity_modelling.cvm.geometry import SmoothingBoundary

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
                raise KeyError(f"Basin {basin_name} not found")

            if "smoothing" in basin:
                boundary_vec_filename = self.get_full_path(basin["smoothing"])
                if boundary_vec_filename.exists():
                    self.log(
                        f"Loading smoothing boundary: {boundary_vec_filename}",
                        VMLogger.INFO,
                    )
                    try:
                        data = np.fromfile(boundary_vec_filename, dtype=float, sep=" ")
                        smooth_lons, smooth_lats = data[0::2], data[1::2]
                        smooth_bound_lons.extend(smooth_lons)
                        smooth_bound_lats.extend(smooth_lats)
                        count += len(smooth_lons)
                        self.log(
                            f"Added {len(smooth_lons)} smoothing points for basin {basin_name}",
                            VMLogger.DEBUG,
                        )
                    except Exception as e:
                        if isinstance(e, (SystemExit, KeyboardInterrupt)):
                            raise
                        self.log(
                            f"Error reading smoothing boundary: {e}", VMLogger.ERROR
                        )
                        raise RuntimeError(
                            f"Failed to load smoothing boundary from {boundary_vec_filename}: {str(e)}"
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
