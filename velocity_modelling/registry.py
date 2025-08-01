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

import logging
from logging import Logger
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import yaml

from velocity_modelling.constants import (
    DEFAULT_OFFSHORE_1D_MODEL,
    DEFAULT_OFFSHORE_DISTANCE,
    MODEL_VERSIONS_ROOT,
    NZCVM_REGISTRY_PATH,
    VelocityTypes,
)
from velocity_modelling.geometry import SmoothingBoundary
from velocity_modelling.global_model import GlobalSurfaceRead


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
    global_params : dict
        Global parameters for the specified model version.
    logger : Logger
        Logger for logging information and errors.
    cache : dict
        Cache for storing loaded data to prevent redundant file operations.


    Parameters
    ----------
    version : str
        The version of the velocity model.
    data_root : Path
        Root directory for the velocity model data.
    registry_path : Path, optional
        Path to the registry YAML file.
    logger : Logger, optional
        Logger instance for logging; created if not provided.


    Raises
    ------
    ValueError
        If the model version file is not found or cannot be loaded.

    """

    def __init__(
        self,
        version: str,
        data_root: Path,
        registry_path: Optional[Path] = NZCVM_REGISTRY_PATH,
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the CVMRegistry.

        """
        # Initialize Logger if not provided
        self.logger = (
            logger if logger is not None else Logger(name="velocity_model.registry")
        )

        self.data_root = data_root

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

        # Check if posInfSurf and negInfSurf paths are present
        has_pos_inf = any(
            s.get("path") == "global/surface/posInf.in" for s in global_surfaces_list
        )
        has_neg_inf = any(
            s.get("path") == "global/surface/negInf.in" for s in global_surfaces_list
        )

        # Add posInfSurf and negInfSurf if needed
        if not has_pos_inf:
            global_surfaces_list.insert(
                0, {"path": "global/surface/posInf.in", "submodel": "nan_submod"}
            )
        if not has_neg_inf:
            global_surfaces_list.append(
                {"path": "global/surface/negInf.in", "submodel": None}
            )

        self.global_params["surfaces"] = global_surfaces_list

        # Create unique identifiers for surfaces (using paths)
        self.global_params["surface_paths"] = [
            Path(d.get("path")) for d in self.global_params["surfaces"]
        ]
        self.global_params["submodels"] = [
            d.get("submodel") for d in self.global_params["surfaces"]
        ][:-1]  # Exclude the last submodel as before

        self.cache = {}  # Initialize cache

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
            self.logger.log(logging.ERROR, f"Error: {datatype} not found in registry")
            return None

        for info in data_section:
            assert "name" in info, f"Error: Entry in {datatype} lacks a 'name' field."
            if info["name"] == name:
                return info
        self.logger.log(
            logging.ERROR,
            f"Error: {name} for datatype {datatype} not found in registry",
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
            self.data_root / relative_path
            if not Path(relative_path).is_absolute()
            else Path(relative_path)
        )

    def load_vm1d_submodel(self, vm1d_path: Path):
        """
        Load a 1D velocity submodel into memory.

        Parameters
        ----------
        vm1d_path : Path or str
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
        from velocity_modelling.velocity1d import VelocityModel1D

        vm1d_path = self.get_full_path(vm1d_path)
        if vm1d_path in self.cache:
            self.logger.log(logging.DEBUG, f"{vm1d_path} loaded from CACHE")
            return self.cache[vm1d_path]

        try:
            vm1d_data = VelocityModel1D(vm1d_path)
            self.cache[vm1d_path] = vm1d_data
            self.logger.log(logging.INFO, f"Loaded 1D velocity model from {vm1d_path}")
            return vm1d_data
        except FileNotFoundError:
            self.logger.log(
                logging.ERROR, f"Error: 1D velocity model file {vm1d_path} not found."
            )
            raise FileNotFoundError(f"1D velocity model file {vm1d_path} not found")
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise  # Re-raise critical exceptions
            self.logger.log(logging.ERROR, f"Error loading 1D velocity model: {e}")
            raise RuntimeError(
                f"Failed to load 1D velocity model from {vm1d_path}: {str(e)}"
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
        from velocity_modelling.basin_model import BasinData

        all_basin_data = []
        for basin_name in basin_names:
            self.logger.log(logging.INFO, f"Loading basin data for {basin_name}")
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
                self.logger.log(
                    logging.ERROR,
                    f"Error: Basin boundary {basin_boundary_path} is not closed.",
                )
                raise ValueError(f"Basin boundary {basin_boundary_path} is not closed")
            return boundary_data
        except FileNotFoundError:
            self.logger.log(
                logging.ERROR,
                f"Error: Basin boundary file {basin_boundary_path} not found.",
            )
            raise FileNotFoundError(
                f"Basin boundary file {basin_boundary_path} not found"
            )
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
            self.logger.log(
                logging.ERROR,
                f"Error reading basin boundary file {basin_boundary_path}: {e}",
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
            self.logger.log(
                logging.ERROR, f"Error: Submodel {submodel_name} not found."
            )
            raise KeyError(f"Submodel {submodel_name} not found")

        if submodel["type"] == "vm1d":
            vm1d = self.get_info("submodel", submodel["name"])
            if vm1d is None:
                self.logger.log(
                    logging.ERROR, f"Error: vm1d {submodel['name']} not found."
                )
                raise KeyError(f"vm1d {submodel['name']} not found")
            if vm1d["type"] != "vm1d":
                self.logger.log(
                    logging.ERROR, f"Error: {submodel['name']} is not a 1D model."
                )
                raise KeyError(f"{submodel['name']} is not a 1D model")
            return (submodel_name, self.load_vm1d_submodel(vm1d["data"]))
        elif submodel["type"] in {"relation", "perturbation"}:
            self.logger.log(
                logging.DEBUG,
                f"Using {submodel['type']} submodel {submodel_name} with no additional data",
            )
            return (submodel_name, None)
        return None

    def load_basin_surface(self, basin_surface: dict):
        """
        Load a basin surface from a file.

        Parameters
        ----------
        basin_surface : dict
            Dictionary with keys 'path' and optionally 'submodel' specifying the basin surface.

        Returns
        -------
        BasinSurfaceRead
            The loaded basin surface data.

        Raises
        ------
        KeyError
            If the basin_surface lacks a path field.
        FileNotFoundError
            If the surface file cannot be found.
        RuntimeError
            If the file cannot be read or parsed correctly.
        """

        try:
            surface_path = basin_surface["path"]
        except KeyError:
            self.logger.log(
                logging.ERROR, "Error: Basin surface lacks required 'path' field."
            )
            raise KeyError("Basin surface definition must include 'path' field")

        self.logger.log(logging.DEBUG, f"Loading basin surface file {surface_path}")
        basin_surface_path = self.get_full_path(surface_path)
        if basin_surface_path in self.cache:
            self.logger.log(logging.DEBUG, f"{basin_surface_path} loaded from cache")
            return self.cache[basin_surface_path]

        try:
            # Check file extension to determine loading method
            if basin_surface_path.suffix.lower() == ".h5":
                basin_surf_read = self._load_hdf5_basin_surface(basin_surface_path)
            else:  # Default to ASCII format
                basin_surf_read = self._load_ascii_basin_surface(basin_surface_path)

            self.cache[basin_surface_path] = basin_surf_read
            return basin_surf_read

        except FileNotFoundError:
            self.logger.log(
                logging.ERROR,
                f"Error: Basin surface file {basin_surface_path} not found.",
            )
            raise FileNotFoundError(
                f"Basin surface file {basin_surface_path} not found"
            )
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
            self.logger.log(logging.ERROR, f"Error loading basin surface: {e}")
            raise RuntimeError(
                f"Failed to load basin surface from {basin_surface_path}: {str(e)}"
            )

    def _load_ascii_basin_surface(self, surface_path: Path):
        """
        Load a basin surface from an ASCII file.

        Parameters
        ----------
        surface_path : Path
            Path to the ASCII surface file.

        Returns
        -------
        BasinSurfaceRead
            The loaded basin surface data.
        """
        from velocity_modelling.basin_model import BasinSurfaceRead

        with open(surface_path, "r") as f:
            nlat, nlon = map(int, f.readline().split())

            latitudes = np.fromfile(f, dtype=float, count=nlat, sep=" ")
            longitudes = np.fromfile(f, dtype=float, count=nlon, sep=" ")

            raster_data = np.fromfile(f, dtype=float, count=nlat * nlon, sep=" ")
            if len(raster_data) != nlat * nlon:
                self.logger.log(
                    logging.WARNING,
                    f"In {surface_path}: Data length mismatch - got {len(raster_data)}, expected {nlat * nlon}. "
                    "Missing data will be padded with 0.",
                )
                raster_data = np.pad(
                    raster_data, (0, nlat * nlon - len(raster_data)), "constant"
                )

            basin_surf_read = BasinSurfaceRead(surface_path, latitudes,longitudes, raster_data.reshape((nlat, nlon)).T)

            return basin_surf_read

    def _load_hdf5_basin_surface(self, surface_path: Path):
        """
        Load a basin surface from an HDF5 file.

        Parameters
        ----------
        surface_path : Path
            Path to the HDF5 surface file.

        Returns
        -------
        BasinSurfaceRead
            The loaded basin surface data.
        """
        from velocity_modelling.basin_model import BasinSurfaceRead

        with h5py.File(surface_path, "r") as f:
            # Read attributes
            basin_surf_read = BasinSurfaceRead(surface_path,  f["latitude"][:],  f["longitude"][:],  f["elevation"][:].T)

        return basin_surf_read

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

        if surface_file is None:
            return None

        surface_path = self.get_full_path(surface_file)

        if surface_path in self.cache:
            self.logger.log(logging.DEBUG, f"{surface_path} loaded from cache")
            return self.cache[surface_path]

        try:
            # Check file extension to determine loading method
            if surface_path.suffix.lower() == ".h5":
                surface_data = self._load_hdf5_global_surface(surface_path)
            else:  # Default to ASCII format
                surface_data = self._load_ascii_global_surface(surface_path)

            self.cache[surface_path] = surface_data
            return surface_data

        except FileNotFoundError:
            self.logger.log(logging.ERROR, f"Surface file {surface_path} not found")
            raise FileNotFoundError(f"Surface file {surface_path} not found")
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
            self.logger.log(
                logging.ERROR, f"Error reading surface file {surface_path}: {e}"
            )
            raise RuntimeError(
                f"Failed to load global surface from {surface_path}: {str(e)}"
            )

    def _load_ascii_global_surface(self, surface_path: Path):
        """
        Load a global surface from an ASCII file.

        Parameters
        ----------
        surface_path : Path
            Path to the ASCII surface file.

        Returns
        -------
        GlobalSurfaceRead
            The loaded global surface data.
        """
        from velocity_modelling.global_model import GlobalSurfaceRead

        with open(surface_path, "r") as f:
            nlat, nlon = map(int, f.readline().split())
            self.logger.log(
                logging.DEBUG,
                f"Reading surface with dimensions {nlat}x{nlon} from {surface_path}",
            )

            latitudes = np.fromfile(f, dtype=float, count=nlat, sep=" ")
            longitudes = np.fromfile(f, dtype=float, count=nlon, sep=" ")
            raster_data = np.fromfile(f, dtype=float, count=nlat * nlon, sep=" ")

            if len(raster_data) != nlat * nlon:
                self.logger.log(
                    logging.WARNING,
                    f"In {surface_path}: Data length mismatch - got {len(raster_data)}, expected {nlat * nlon}. "
                    "Missing data will be padded with 0.",
                )
                raster_data = np.pad(
                    raster_data, (0, nlat * nlon - len(raster_data)), "constant"
                )

            return GlobalSurfaceRead(surface_path, latitudes, longitudes, raster_data.reshape((nlat, nlon)).T
            )

    def _load_hdf5_global_surface(self, surface_path: Path):
        """
        Load a global surface from an HDF5 file.

        Parameters
        ----------
        surface_path : Path
            Path to the HDF5 surface file.

        Returns
        -------
        GlobalSurfaceRead
            The loaded global surface data.
        """
        from velocity_modelling.global_model import GlobalSurfaceRead

        with h5py.File(surface_path, "r") as f:
            # Read datasets
            latitudes = f["latitude"][:]
            longitudes = f["longitude"][:]
            # HDF5 stores in row-major order, so transpose as needed
            elevation_data = f["elevation"][:].T

            self.logger.log(
                logging.DEBUG,
                f"Reading HDF5 surface with dimensions {len(latitudes)}x{len(longitudes)} from {surface_path}",
            )

            return GlobalSurfaceRead(surface_path, latitudes, longitudes, elevation_data)

    def load_tomo_surface_data(
        self,
        tomo_name: str,
        offshore_surface_path: Optional[Path] = DEFAULT_OFFSHORE_DISTANCE,
        offshore_v1d_name: Optional[str] = DEFAULT_OFFSHORE_1D_MODEL,
    ):
        """
        Load tomography surface data from registry.

        Parameters
        ----------
        tomo_name : str
            The name of the tomography data in the registry.
        offshore_surface_path : Path, optional
            The path to the offshore distance surface.
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
        from velocity_modelling.global_model import TomographyData

        tomo = self.get_info("tomography", tomo_name)
        if tomo is None:
            raise KeyError(f"Tomography data {tomo_name} not found")

        surf_depth = tomo["elev"]
        special_offshore_tapering = tomo["special_offshore_tapering"]
        vs30 = self.load_global_surface(tomo["vs30_path"])

        # Determine format based on the path rather than an explicit format attribute
        tomo_path = self.get_full_path(tomo["path"])
        if tomo_path.is_file() and tomo_path.suffix == ".h5":
            data_format = "HDF5"
        else:
            data_format = "ASCII"

        self.logger.log(
            logging.INFO,
            f"Loading tomography data '{tomo_name}' ({data_format} format) with {len(surf_depth)} depth levels",
        )

        # Load surfaces based on data format
        if data_format == "HDF5":
            surfaces = self._load_hdf5_tomo_surface_data(
                tomo["path"], tomo_name, surf_depth
            )
        else:  # ASCII
            surfaces = self._load_ascii_tomo_surface_data(tomo["path"], surf_depth)

        # Load offshore data (unchanged)
        self.logger.log(
            logging.INFO,
            f"Loading offshore distance surface: {offshore_surface_path}",
        )
        offshore_distance_surface = self.load_global_surface(offshore_surface_path)

        offshore_v1d_info = self.get_info("submodel", offshore_v1d_name)
        if offshore_v1d_info is None:
            self.logger.log(
                logging.ERROR,
                f"Error: Offshore 1D model {offshore_v1d_name} not found",
            )
            raise KeyError(f"Offshore 1D model {offshore_v1d_name} not found")
        if offshore_v1d_info["type"] != "vm1d":
            self.logger.log(
                logging.ERROR,
                f"Error: Offshore 1D model {offshore_v1d_name} is not a 1D model",
            )
            raise KeyError(f"Offshore 1D model {offshore_v1d_name} is not a 1D model")
        offshore_basin_model_1d = self.load_vm1d_submodel(offshore_v1d_info["data"])

        return TomographyData(
            name=tomo_name,
            surf_depth=surf_depth,
            special_offshore_tapering=special_offshore_tapering,
            vs30=vs30,
            surfaces=surfaces,
            offshore_distance_surface=offshore_distance_surface,
            offshore_basin_model_1d=offshore_basin_model_1d,
        )

    def _load_hdf5_tomo_surface_data(
        self, path: Path, tomo_name: str, surf_depth: list
    ) -> list[dict[str, GlobalSurfaceRead]]:
        """
        Load tomography surfaces from an HDF5 file.

        Parameters
        ----------
        path : Path
            The relative path to the HDF5 file.
        tomo_name : str
            The name of the tomography data.
        surf_depth : list
            List of elevation depths for the tomography surfaces.

        Returns
        -------
        list[dict[str, GlobalSurfaceRead]]
            List of dictionaries containing GlobalSurfaceRead objects per velocity type (ie. vp, vs, rho) for each elevation.
        """
        surfaces = []
        hdf5_path = self.get_full_path(path)
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 tomography file {hdf5_path} not found")

        self.logger.log(
            logging.INFO, f"Loading tomography surfaces from HDF5: {hdf5_path}"
        )
        try:
            with h5py.File(hdf5_path, "r") as h5f:
                for i, elev in enumerate(surf_depth):
                    surfaces.append({})
                    elev_str = str(int(elev)) if elev == int(elev) else f"{elev:.2f}"

                    try:
                        elev_group = h5f[elev_str]
                        latitudes = elev_group["latitudes"][:]
                        longitudes = elev_group["longitudes"][:]

                        for vtype in VelocityTypes:
                            try:
                                data = elev_group[vtype.name][:]
                                surfaces[i][vtype.name] = GlobalSurfaceRead(hdf5_path,
                                    latitudes, longitudes, data.T
                                )
                                self.logger.log(
                                    logging.DEBUG,
                                    f"Loaded tomography surface for {vtype.name} at elevation {elev}",
                                )
                            except KeyError:
                                self.logger.log(
                                    logging.ERROR,
                                    f"Error: Data for {vtype.name} at elevation {elev} not found in HDF5",
                                )
                                raise FileNotFoundError(
                                    f"Tomography data for {vtype.name} at elevation {elev} not found"
                                )
                    except KeyError:
                        self.logger.log(
                            logging.ERROR, f"Error: Elevation {elev} not found in HDF5"
                        )
                        raise FileNotFoundError(f"Elevation {elev} not found in HDF5")
        except (IOError, OSError, RuntimeError) as e:
            self.logger.log(
                logging.ERROR, f"Error opening or reading HDF5 file {hdf5_path}: {e}"
            )
            raise RuntimeError(f"Failed to load HDF5 file {hdf5_path}: {str(e)}")

        return surfaces

    def _load_ascii_tomo_surface_data(
        self, path: Path, surf_depth: list
    ) -> list[dict[str, GlobalSurfaceRead]]:
        """
        Load tomography surfaces from ASCII files.

        Parameters
        ----------
        path : Path
            The relative path to the ASCII files.
        surf_depth : list
            List of elevation depths for the tomography surfaces.

        Returns
        -------
        list[dict[str, GlobalSurfaceRead]]
            List of dictionaries containing GlobalSurfaceRead objects per velocity type (ie. vp, vs, rho) for each elevation.

        """
        surfaces = []
        base_path = self.get_full_path(path)

        self.logger.log(
            logging.INFO, f"Loading tomography surfaces from ASCII files in {base_path}"
        )
        for i, elev in enumerate(surf_depth):
            surfaces.append({})
            elev_name = (
                f"{elev}" if elev == int(elev) else f"{elev:.2f}".replace(".", "p")
            )

            for vtype in VelocityTypes:
                tomofile = (
                    base_path / f"surf_tomography_{vtype.name}_elev{elev_name}.in"
                )
                if not tomofile.exists():
                    self.logger.log(
                        logging.ERROR, f"Error: Tomography file {tomofile} not found"
                    )
                    raise FileNotFoundError(
                        f"Tomography file {tomofile} does not exist"
                    )

                surfaces[i][vtype.name] = self.load_global_surface(tomofile)
                self.logger.log(
                    logging.DEBUG,
                    f"Loaded tomography surface for {vtype.name} at elevation {elev}",
                )

        return surfaces

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
        vm1d_data = None
        tomography_data = None

        self.logger.log(logging.INFO, "Loading global surfaces")
        global_surfaces = self.load_global_surface_data(
            self.global_params["surface_paths"]
        )

        self.logger.log(logging.INFO, "Loading global velocity submodel data")
        for submodel_name in self.global_params["submodels"]:
            submodel_info = self.get_info("submodel", submodel_name)
            if submodel_info is None:
                self.logger.log(
                    logging.ERROR, f"Error: Submodel {submodel_name} not found"
                )
                raise KeyError(f"Submodel {submodel_name} not found")

            if submodel_info["type"] is None:
                self.logger.log(
                    logging.DEBUG, "nan submodel recognized (no data to load)"
                )
            elif submodel_info["type"] == "vm1d":
                vm1d_data = self.load_vm1d_submodel(submodel_info["data"])
                self.logger.log(logging.INFO, "Loaded 1D velocity model data")
            elif submodel_info["type"] == "tomography":
                if self.global_params.get("tomography"):
                    tomography_data = self.load_tomo_surface_data(
                        self.global_params["tomography"]
                    )
                else:
                    self.logger.log(logging.ERROR, "Error: Tomography data not found")
                    raise KeyError("Tomography data not found")
            elif submodel_info["type"] == "relation":
                self.logger.log(
                    logging.DEBUG,
                    f"Using relation submodel {submodel_name} with no additional data",
                )
            else:
                self.logger.log(
                    logging.INFO,
                    f"Error: Unknown submodel type {submodel_info['type']} to be ignored.",
                )

        if tomography_data:
            self.logger.log(
                logging.INFO,
                "Loading smooth boundaries for tomography transitions",
            )
            tomography_data.smooth_boundary = self.load_smooth_boundaries(
                self.global_params["basins"]
            )

        self.logger.log(logging.INFO, "Loading basin data")
        basin_data = self.load_basin_data(self.global_params["basins"])

        self.logger.log(logging.INFO, "All global data successfully loaded")
        return vm1d_data, tomography_data, global_surfaces, basin_data

    def load_global_surface_data(
        self, global_surface_paths: list[Path]
    ) -> list[GlobalSurfaceRead]:
        """
        Load multiple global surfaces from the registry.

        Parameters
        ----------
        global_surface_paths : list[Path]
            List of global surface paths defined in the registry.

        Returns
        -------
        list[GlobalSurfaceRead]
            List of GlobalSurfaceRead objects containing the loaded surface data.
        """
        surfaces = []
        for surface_path in global_surface_paths:
            self.logger.log(
                logging.DEBUG, f"Loading global surface: {surface_path.name}"
            )
            surfaces.append(self.load_global_surface(surface_path))

        self.logger.log(
            logging.INFO, f"Loaded {len(global_surface_paths)} global surfaces"
        )
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
        from velocity_modelling.geometry import SmoothingBoundary

        smooth_bound_lons = []
        smooth_bound_lats = []
        count = 0

        self.logger.log(
            logging.INFO,
            f"Loading smoothing boundaries for {len(basin_names)} basins",
        )
        for basin_name in basin_names:
            basin = self.get_info("basin", basin_name)
            if basin is None:
                self.logger.log(
                    logging.ERROR, f"Basin {basin_name} not found in registry"
                )
                raise KeyError(f"Basin {basin_name} not found")

            if "smoothing" in basin:
                boundary_vec_filename = self.get_full_path(basin["smoothing"])
                if boundary_vec_filename.exists():
                    self.logger.log(
                        logging.INFO,
                        f"Loading smoothing boundary: {boundary_vec_filename}",
                    )
                    try:
                        data = np.fromfile(boundary_vec_filename, dtype=float, sep=" ")
                        smooth_lons, smooth_lats = data[0::2], data[1::2]
                        smooth_bound_lons.extend(smooth_lons)
                        smooth_bound_lats.extend(smooth_lats)
                        count += len(smooth_lons)
                        self.logger.log(
                            logging.DEBUG,
                            f"Added {len(smooth_lons)} smoothing points for basin {basin_name}",
                        )
                    except Exception as e:
                        if isinstance(e, (SystemExit, KeyboardInterrupt)):
                            raise
                        self.logger.log(
                            logging.ERROR, f"Error reading smoothing boundary: {e}"
                        )
                        raise RuntimeError(
                            f"Failed to load smoothing boundary from {boundary_vec_filename}: {str(e)}"
                        )
                else:
                    self.logger.log(
                        logging.WARNING,
                        f"Smoothing boundary file not found: {boundary_vec_filename} -- to be ignored",
                    )
            else:
                self.logger.log(
                    logging.DEBUG, f"No smoothing defined for basin {basin_name}"
                )

        self.logger.log(logging.INFO, f"Total smoothing boundary points: {count}")
        return SmoothingBoundary(smooth_bound_lons, smooth_bound_lats)
