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
import re
from logging import Logger
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import h5py
import numpy as np
import yaml

from velocity_modelling.constants import (
    DEFAULT_OFFSHORE_1D_MODEL,
    DEFAULT_OFFSHORE_DISTANCE,
    MODEL_VERSIONS_ROOT,
    VelocityTypes,
    get_registry_path,
)
from velocity_modelling.geometry import SmoothingBoundary
from velocity_modelling.global_model import GlobalSurfaceRead


def get_all_basins_dict(registry_path: Path):
    """
    Reads and parses the registry yaml file to group basin info into a dictionary.by basin name.

    Parameters
    ----------
    registry_path : Path
        Path to the nzcvm_registry.yaml file.

    Returns
    -------
    dict
        A dictionary where keys are basin names and values are lists of dictionaries
        containing basin details, including full name, version, version tuple, and data.

    Raises
    ------
    FileNotFoundError
        If the registry file is not found.
    """
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found at {registry_path}")

    with open(registry_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    basin_versions = {}
    for basin_item in data.get("basin", []):
        full_name = basin_item.pop("name")
        match = re.match(r"^(.*?)_v(\d+p\d+)$", full_name)
        if not match:
            continue
        basin_name, version = match.groups()
        version_parts = version.replace("v", "").split("p")
        version_tuple = (int(version_parts[0]), int(version_parts[1]))

        if basin_name not in basin_versions:
            basin_versions[basin_name] = []
        basin_versions[basin_name].append(
            {
                "full_name": full_name,
                "version": version,
                "version_tuple": version_tuple,
                "data": basin_item,
            }
        )

    return basin_versions


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
        registry_path: Optional[Path] = None,
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

        if not registry_path:
            registry_path = get_registry_path()
        try:
            with open(registry_path, "r") as f:
                self.registry = yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.log(
                logging.ERROR, f"Error: Registry file {registry_path} not found."
            )
            raise FileNotFoundError(f"Registry file {registry_path} not found")

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

        # Ensure 'basins' is always a list, defaulting to empty if missing or None
        self.global_params["basins"] = self.global_params.get("basins") or []

        # Validate global_params
        global_surfaces_list = self.global_params.get("surfaces", [])
        if not global_surfaces_list:
            global_surfaces_list = []

        # Check if posInfSurf and negInfSurf paths are present
        has_pos_inf = any(
            s.get("path") == "surface/posInf.h5" for s in global_surfaces_list
        )
        has_neg_inf = any(
            s.get("path") == "surface/negInf.h5" for s in global_surfaces_list
        )

        # Add posInfSurf and negInfSurf if needed
        if not has_pos_inf:
            global_surfaces_list.insert(
                0, {"path": "surface/posInf.h5", "submodel": "nan_submod"}
            )
        if not has_neg_inf:
            global_surfaces_list.append({"path": "surface/negInf.h5", "submodel": None})

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

    def load_basin_boundary(self, basin_boundary_path: Path | str) -> np.ndarray:
        """
        Load a basin boundary from a .txt (lon lat per line) or .geojson file.

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
        RuntimeError
            If the file cannot be read or parsed correctly.
        """
        path = self.get_full_path(basin_boundary_path)
        if not path.exists():
            raise FileNotFoundError(f"Basin boundary file {path} not found")

        try:
            if path.suffix.lower() in (".geojson", ".json"):
                return self._load_geojson_boundary(path)
            else:
                return self._load_ascii_boundary(path)
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
            self.logger.log(logging.ERROR, f"Error reading basin boundary {path}: {e}")
            raise RuntimeError(f"Failed to load basin boundary from {path}: {e}")

    def _load_geojson_boundary(self, path: Path) -> np.ndarray:
        """
        Load boundary from GeoJSON file.

        Parameters
        ----------
        path : Path
            Path to the GeoJSON file containing boundary data.

        Returns
        -------
        np.ndarray
            Array of boundary coordinates as [longitude, latitude] pairs.

        Raises
        ------
        ValueError
            If the GeoJSON contains unsupported geometry types or no coordinates.
        """
        gdf = gpd.read_file(path)

        parts = []
        for geom in gdf.geometry:
            if geom is None:
                continue

            # Extract coordinate rings based on geometry type
            if geom.geom_type == "Polygon":
                rings = [geom.exterior]
            elif geom.geom_type == "MultiPolygon":
                rings = [poly.exterior for poly in geom.geoms]
            elif geom.geom_type in ("LineString", "MultiLineString"):
                rings = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
            else:
                raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

            # Convert rings to coordinates and ensure closure
            for ring in rings:
                coords = np.asarray(ring.coords, dtype=float)
                if not np.allclose(coords[0], coords[-1], equal_nan=False):
                    coords = np.vstack([coords, coords[0]])
                parts.extend([coords, np.array([[np.nan, np.nan]])])

        if not parts:
            raise ValueError("No coordinates found in GeoJSON")

        return np.vstack(parts[:-1])  # Remove trailing separator

    def _load_ascii_boundary(self, path: Path) -> np.ndarray:
        """
        Load boundary from ASCII file.
        This function will be deprecated in future versions in favor of GeoJSON.

        Parameters
        ----------
        path : Path
            Path to the ASCII file with 'lon lat' format per line.

        Returns
        -------
        np.ndarray
            Array of boundary coordinates as [longitude, latitude] pairs.

        Raises
        ------
        ValueError
            If the file format is invalid (not 'lon lat' per line).
        """
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Invalid format in {path}, expected 'lon lat' per line")

        boundary = data[:, :2]  # Take only first two columns
        # Ensure closure
        if not np.allclose(boundary[0], boundary[-1], equal_nan=False):
            boundary = np.vstack([boundary, boundary[0]])

        return boundary

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
            return submodel_name, self.load_vm1d_submodel(vm1d["data"])
        elif submodel["type"] in ["relation", "perturbation"]:
            self.logger.log(
                logging.DEBUG,
                f"Using {submodel['type']} submodel {submodel_name} with no additional data",
            )
            return submodel_name, None
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
        This function will be deprecated in future versions in favor of HDF5.

        Parameters
        ----------
        surface_path : Path
            Path to the ASCII surface file.

        Returns
        -------
        BasinSurfaceRead
            The loaded basin surface data.

        Raises
        ------
        IOError
            If the ASCII file cannot be read.
        ValueError
            If the file format is incorrect or data is malformed.
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

            basin_surf_read = BasinSurfaceRead(
                surface_path, latitudes, longitudes, raster_data.reshape((nlat, nlon)).T
            )

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

        Raises
        ------
        IOError
            If the HDF5 file cannot be read.
        KeyError
            If required datasets are missing from the HDF5 file.
        """
        from velocity_modelling.basin_model import BasinSurfaceRead

        with h5py.File(surface_path, "r") as f:
            # Read attributes
            basin_surf_read = BasinSurfaceRead(
                surface_path, f["latitude"][:], f["longitude"][:], f["elevation"][:].T
            )

        return basin_surf_read

    def load_global_surface(self, surface_file: Path | str) -> GlobalSurfaceRead | None:
        """
        Load a global surface raster from a file.

        Parameters
        ----------
        surface_file : Path or str
            Path to the global surface file.

        Returns
        -------
        GlobalSurfaceRead or None
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
        This function will be deprecated in future versions in favor of HDF5.

        Parameters
        ----------
        surface_path : Path
            Path to the ASCII surface file.

        Returns
        -------
        GlobalSurfaceRead
            The loaded global surface data.

        Raises
        ------
        IOError
            If the ASCII file cannot be read.
        ValueError
            If the file format is incorrect or data dimensions don't match.
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

            return GlobalSurfaceRead(
                surface_path, latitudes, longitudes, raster_data.reshape((nlat, nlon)).T
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

        Raises
        ------
        IOError
            If the HDF5 file cannot be read.
        KeyError
            If required datasets are missing from the HDF5 file.
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

            return GlobalSurfaceRead(
                surface_path, latitudes, longitudes, elevation_data
            )

    def load_tomo_surface_data(
        self,
        tomo_cfg: dict,
        offshore_surface_path: Path = DEFAULT_OFFSHORE_DISTANCE,
        offshore_v1d_name: str = DEFAULT_OFFSHORE_1D_MODEL,
    ):
        """
         Load tomography surfaces by combining base info from the registry (intrinsic)
        with per-version overrides from the model version file (operational params).
        Expected tomography entry example (from model version yaml):
          - name: EP2010
            vs30: nz_with_offshore
            special_offshore_tapering: true
            GTL: true

        Parameters
        ----------
        tomo_cfg : dict
            Configuration dictionary for the tomography data, containing:

        offshore_surface_path : Path, optional, default=DEFAULT_OFFSHORE_DISTANCE
            The path to the offshore distance surface.
        offshore_v1d_name : str, optional, default=DEFAULT_OFFSHORE_1D_MODEL
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

        # 1) intrinsic info from registry (name, elev, path, author, title, url)
        tomo_name = tomo_cfg["name"]
        base = self.get_info("tomography", tomo_name)
        if base is None:
            raise KeyError(f"Tomography {tomo_name} not found in registry")

        surf_depth = base["elev"]
        tomo_rel_path = base["path"]

        # 2) per-version options (may be absent; provide sensible defaults)
        special_offshore_tapering = bool(
            tomo_cfg.get("special_offshore_tapering", False)
        )
        vs30_name = tomo_cfg.get("vs30")  # e.g., "nz_with_offshore"
        gtl_enabled = bool(tomo_cfg.get("GTL", False))

        # Resolve vs30 from registry if provided as a named asset
        vs30_surface = None
        if vs30_name is not None:
            vs30_info = self.get_info("vs30", vs30_name)
            if vs30_info is None:
                raise KeyError(f"vs30 '{vs30_name}' not found in registry")
            vs30_surface = self.load_global_surface(vs30_info["path"])

        # Detect data format by file extension
        tomo_path = self.get_full_path(tomo_rel_path)
        data_format = (
            "HDF5"
            if (tomo_path.is_file() and tomo_path.suffix.lower() == ".h5")
            else "ASCII"
        )
        self.logger.log(
            logging.INFO,
            f"Loading tomography '{tomo_name}' ({data_format}) with {len(surf_depth)} levels",
        )

        # Load tomography surfaces
        if data_format == "HDF5":
            surfaces = self._load_hdf5_tomo_surface_data(tomo_rel_path, surf_depth)
        else:
            surfaces = self._load_ascii_tomo_surface_data(tomo_rel_path, surf_depth)

        # Offshore distance & default offshore 1D
        self.logger.log(
            logging.INFO, f"Loading offshore distance surface: {offshore_surface_path}"
        )
        offshore_distance_surface = self.load_global_surface(offshore_surface_path)

        offshore_v1d_info = self.get_info("submodel", offshore_v1d_name)
        if offshore_v1d_info is None or offshore_v1d_info.get("type") != "vm1d":
            raise KeyError(
                f"Offshore 1D model {offshore_v1d_name} not found or not a vm1d"
            )
        offshore_basin_model_1d = self.load_vm1d_submodel(offshore_v1d_info["data"])

        return TomographyData(
            name=tomo_name,
            surf_depth=surf_depth,
            special_offshore_tapering=special_offshore_tapering,
            gtl=gtl_enabled,
            vs30=vs30_surface,
            surfaces=surfaces,
            offshore_distance_surface=offshore_distance_surface,
            offshore_basin_model_1d=offshore_basin_model_1d,
        )

    def _load_hdf5_tomo_surface_data(
        self, path: Path, surf_depth: list
    ) -> list[dict[str, GlobalSurfaceRead]]:
        """
        Load tomography surfaces from an HDF5 file.

        Parameters
        ----------
        path : Path
            The relative path to the HDF5 file.
        surf_depth : list
            List of elevation depths for the tomography surfaces.

        Returns
        -------
        list[dict[str, GlobalSurfaceRead]]
            List of dictionaries containing GlobalSurfaceRead objects per velocity type (ie. vp, vs, rho) for each elevation.

        Raises
        ------
        FileNotFoundError
            If the HDF5 file or required elevation data is not found.
        RuntimeError
            If the HDF5 file cannot be opened or read.
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
                                surfaces[i][vtype.name] = GlobalSurfaceRead(
                                    hdf5_path, latitudes, longitudes, data.T
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
        This function will be deprecated in future versions in favor of HDF5.

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

        Raises
        ------
        FileNotFoundError
            If any required tomography ASCII files are missing.
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
                # New: self.global_params["tomography"] can be a list of dicts
                tomo_list = self.global_params.get("tomography", [])
                if not tomo_list:
                    self.logger.log(
                        logging.ERROR,
                        "Error: Tomography config missing in version file",
                    )
                    raise KeyError("Tomography config missing in version file")

                # For now, if your engine expects a single TomographyData, use the first.
                # If you plan multi-tomography mixes, store the list and adapt downstream.
                tomography_data = self.load_tomo_surface_data(tomo_list[0])
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
