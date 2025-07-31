import logging
import sys
import time

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Annotated, Optional

import typer
from tqdm import tqdm

from qcore import cli
from velocity_modelling.basin_model import (
    InBasin,

    PartialBasinSurfaceDepths,
)
from velocity_modelling.constants import (
    DATA_ROOT,
    NZCVM_REGISTRY_PATH,
    TopoTypes,
)
from velocity_modelling.geometry import (
    extract_mesh_vector,
    extract_partial_mesh,
    gen_full_model_grid_great_circle,
)
from velocity_modelling.global_model import PartialGlobalSurfaceDepths
from velocity_modelling.registry import CVMRegistry

from velocity_modelling.velocity3d import QualitiesVector



# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nzcvm")

app = typer.Typer(pretty_exceptions_enable=False)


def read_depth_points_text_file(file_path: Path, logger: logging.Logger) -> list[float]:
    """
    Read depth points from a text file using numpy.loadtxt.

    The file is expected to contain one depth value (in kilometers) per line.

    Parameters
    ----------
    file_path : Path
        Path to the text file containing depth points.
    logger : logging.Logger
        Logger instance for logging messages.

    Returns
    -------
    List[float]
        List of depth values (in kilometers).

    Raises
    ------
    OSError
        If the file cannot be read or does not exist.
    ValueError
        If the file contains invalid data.
    """
    try:
        if not file_path.exists():
            logger.error(f"Depth points file does not exist: {file_path}")
            raise OSError(f"Depth points file does not exist: {file_path}")

        depth_values = np.loadtxt(file_path, dtype=float).tolist()
        if not depth_values:
            logger.error(f"No valid depth points found in {file_path}")
            raise ValueError(f"No valid depth points found in {file_path}")

        logger.debug(f"Read {len(depth_values)} depth points from {file_path}")
        return depth_values

    except OSError as e:
        logger.error(f"Failed to read depth points file {file_path}: {e}")
        raise OSError(f"Failed to read depth points file {file_path}: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid data in depth points file {file_path}: {e}")
        raise ValueError(f"Invalid data in depth points file {file_path}: {str(e)}")


def write_profiles(
        out_dir: Path,
        qualities_vector: QualitiesVector,
        vm_params: dict,
        mesh_vector: QualitiesVector,
        df: pd.DataFrame,
        profile_idx: int,
        logger: logging.Logger,
) -> None:
    """
    Write velocity profile data to a text file in either 1D_SITE_RESPONSE or STANDARD format.

    Parameters
    ----------
    out_dir : Path
        Output directory where the Profiles subdirectory will be created.
    qualities_vector : QualitiesVector
        Velocity and density data (Vp, Vs, Rho) for the profile.
    vm_params : dict
        Configuration parameters, including model_version, topo_type, min_vs, and output_type.
    mesh_vector : QualitiesVector
        Mesh data containing depth points (Z), latitude (Lat), and longitude (Lon).
    df : pd.DataFrame
        DataFrame containing profile parameters (id, lon, lat, zmin, zmax, spacing).
    profile_idx : int
        Index of the profile being processed.
    logger : logging.Logger
        Logger instance for logging messages.

    Raises
    ------
    ValueError
        If output_type is not '1D_SITE_RESPONSE' or 'STANDARD'.
    OSError
        If there are issues creating or writing to the output file.
    """
    profiles_dir = out_dir / "Profiles"
    try:
        profiles_dir.mkdir(exist_ok=True, parents=True)
    except OSError as e:
        logger.error(f"Failed to create Profiles directory {profiles_dir}: {e}")
        raise OSError(f"Failed to create Profiles directory {profiles_dir}: {str(e)}")

    output_type = vm_params.get("output_type", "STANDARD")
    profile_id = df['id'].iloc[profile_idx]

    if output_type == "1D_SITE_RESPONSE":
        file_path = profiles_dir / f"{profile_id}.1d"
        try:
            with file_path.open('w') as f:
                f.write(f"{mesh_vector.nZ}\n")
                dep_bot = 0.0
                for i in range(mesh_vector.nZ):
                    vs = max(qualities_vector.Vs[i], vm_params["min_vs"])
                    if i == mesh_vector.nZ - 1:
                        delta_depth = -999999.0
                    elif i == 0:
                        delta_depth = 2 * mesh_vector.Z[i]
                        dep_bot = delta_depth
                    else:
                        delta_depth = 2 * (mesh_vector.Z[i] - dep_bot)
                        dep_bot += delta_depth
                    quality_factor_1 = 2.0 * (41.0 + 34.0 * vs)
                    quality_factor_2 = 41.0 + 34.0 * vs
                    f.write(
                        f"{-delta_depth / 1000:.3f} \t {qualities_vector.Vp[i]:.3f} \t "
                        f"{vs:.3f} \t {qualities_vector.Rho[i]:.3f} \t "
                        f"{quality_factor_1:.3f} \t {quality_factor_2:.3f}\n"
                    )
            logger.info(f"Wrote 1D site response profile to {file_path}")
        except OSError as e:
            logger.error(f"Failed to write profile to {file_path}: {e}")
            raise OSError(f"Failed to write profile to {file_path}: {str(e)}")

    elif output_type == "STANDARD":
        file_path = profiles_dir / f"Profile{profile_id}.txt"
        try:
            with file_path.open('w') as f:
                f.write(
                    f"Properties at Lat: {df['lat'].iloc[profile_idx]:.6f} Lon: {df['lon'].iloc[profile_idx]:.6f}\n")
                f.write("Depth (km) \t Vp (km/s) \t Vs (km/s) \t Rho (t/m^3)\n")
                for i in range(mesh_vector.nz):
                    vs = max(qualities_vector.vs[i], vm_params["min_vs"])
                    f.write(
                        f"{mesh_vector.z[i] / 1000:.6f} \t {qualities_vector.vp[i]:.6f} \t "
                        f"{vs:.6f} \t {qualities_vector.rho[i]:.6f}\n"
                    )
            logger.info(f"Wrote standard profile to {file_path}")
        except OSError as e:
            logger.error(f"Failed to write profile to {file_path}: {e}")
            raise OSError(f"Failed to write profile to {file_path}: {str(e)}")

    else:
        logger.error(f"Invalid output_type: {output_type}")
        raise ValueError(f"Invalid output_type: {output_type}. Must be '1D_SITE_RESPONSE' or 'STANDARD'")


def write_profile_surface_depths(
        out_dir: Path,
        cvm_registry: CVMRegistry,
        basin_data_list: list,
        partial_global_surface_depths: PartialGlobalSurfaceDepths,
        partial_basin_surface_depths: PartialBasinSurfaceDepths,
        in_basin_list: list[InBasin],
        mesh_vector: QualitiesVector,
        df: pd.DataFrame,
        profile_idx: int,
        logger: logging.Logger,
) -> None:
    """
    Write surface depths for global and basin surfaces to a text file.

    Parameters
    ----------
    out_dir : Path
        Output directory where the Profiles subdirectory will be created.
    cvm_registry : CVMRegistry
        Registry containing global model parameters (nSurf, surf, nBasins, basin, basinSurfaceNames).
    basin_data_list : list
        List of basin data objects.
    partial_global_surface_depths : PartialGlobalSurfaceDepths
        Depths of global surfaces at the profile location.
    partial_basin_surface_depths : PartialBasinSurfaceDepths
        Depths of basin surfaces at the profile location.
    in_basin : InBasin
        Object indicating whether the profile is in each basin.
    mesh_vector : QualitiesVector
        Mesh data containing latitude (Lat) and longitude (Lon).
    df : pd.DataFrame
        DataFrame containing profile parameters (id, lon, lat, zmin, zmax, spacing).
    profile_idx : int
        Index of the profile being processed.
    logger : logging.Logger
        Logger instance for logging messages.

    Raises
    ------
    OSError
        If there are issues creating or writing to the output file.
    """
    profiles_dir = out_dir / "Profiles"
    try:
        profiles_dir.mkdir(exist_ok=True, parents=True)
    except OSError as e:
        logger.error(f"Failed to create Profiles directory {profiles_dir}: {e}")
        raise OSError(f"Failed to create Profiles directory {profiles_dir}: {str(e)}")

    file_path = profiles_dir / f"ProfileSurfaceDepths{df['id'].iloc[profile_idx]}.txt"
    lines = [
        f"Surface Depths (in m) at Lat: {df['lat'].iloc[profile_idx]:.6f} Lon: {df['lon'].iloc[profile_idx]:.6f}\n",
        "\nGlobal surfaces\n",
        "Surface_name \t Depth (m)\n",
        *[f"{Path(cvm_registry.global_params['surfaces'][i]['path']).stem}\t{partial_global_surface_depths.depths[i]:.6f}\n"
          for i in range(len(cvm_registry.global_params["surfaces"]))],
        "\nBasin surfaces (if applicable)\n",
    ]

    for i, basin in enumerate(basin_data_list):
        if in_basin_list[i].in_basin_lat_lon:
            lines.append(f"\n{basin.name}\n")
            lines.extend(
                f"{basin.surface_names[j]}\t{partial_basin_surface_depths.depths[i][j]:.6f}\n"
                for j in range(basin.nSurfaces)
            )

    try:
        with file_path.open('w') as f:
            f.writelines(lines)
        logger.info(f"Wrote surface depths to {file_path}")
    except OSError as e:
        logger.error(f"Failed to write surface depths to {file_path}: {e}")
        raise OSError(f"Failed to write surface depths to {file_path}: {str(e)}")

@cli.from_docstring(app)
def generate_multiple_profiles(
    out_dir: Annotated[Path, typer.Option(file_okay=False, help="Output directory for profile files")],
    model_version: Annotated[str, typer.Option(help="Version of the model to use")],
    location_csv: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="CSV file with profile parameters (id, lon, lat, zmin, zmax, spacing)")],
    min_vs: Annotated[float, typer.Option(help="Minimum shear wave velocity")] = 0.0,
    topo_type: Annotated[str, typer.Option(help="Topography type")] = TopoTypes.TRUE.name,
    custom_depth: Annotated[Optional[Path], typer.Option(exists=True, dir_okay=False, help="Text file with custom depth points (overrides zmin, zmax, spacing in location_csv)")] = None,
    nzcvm_registry: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = NZCVM_REGISTRY_PATH,
    data_root: Annotated[Path, typer.Option(file_okay=False, exists=True, help="Override the default DATA_ROOT directory")] = DATA_ROOT,
    log_level: Annotated[str, typer.Option(help="Logging level")] = "INFO",
) -> None:
    """
    Generate multiple 1D velocity profiles based on input coordinates and depth parameters.

    This function orchestrates the generation of multiple velocity profiles:
    1. Reads profile coordinates and depth parameters from a CSV file (id, lon, lat, zmin, zmax, spacing)
    2. Optionally reads custom depth points from a text file to override CSV depth parameters
    3. Loads all required datasets (global models, tomography, basins)
    4. Processes each profile and populates velocity/density values
    5. Writes results to disk

    Parameters
    ----------
    out_dir : Path
        Path to the output directory where profile files will be written.
    model_version : str
        Version of the model to use.
    location_csv : Path
        Path to the CSV file containing profile parameters (id, lon, lat, zmin, zmax, spacing).
    min_vs : float, optional
        Minimum shear wave velocity (default: 0.0).
    topo_type : str, optional
        Topography type (default: TRUE).
    custom_depth : Path, optional
        Path to the text file containing custom depth points (overrides zmin, zmax, spacing in CSV).
    nzcvm_registry : Path, optional
        Path to the model registry file (default: NZCVM_REGISTRY_PATH).
    data_root : Path, optional
        Override the default DATA_ROOT directory (default: derived from constants.py).
    log_level : str, optional
        Logging level for the script (default: "INFO").

    Raises
    ------
    ValueError
        If input parameters are invalid.
    OSError
        If there are issues reading input files or writing to the output directory.
    RuntimeError
        If an error occurs during profile generation or data processing.
    """
    start_time = time.time()

    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.log(logging.DEBUG, f"Logger initialized with level {log_level}")
    logger.log(logging.INFO, "Beginning multiple profiles generation")

    # Validate DATA_ROOT
    data_root = data_root.resolve()
    logger.log(logging.INFO, f"data_root set to {data_root}")


    # Validate min_vs
    if min_vs < 0:
        logger.error(f"min_vs ({min_vs}) cannot be negative")
        raise ValueError(f"min_vs ({min_vs}) cannot be negative")

    # Validate and import the appropriate writer based on format
    try:
        _ = TopoTypes[topo_type.upper()]
    except KeyError:
        logger.log(logging.ERROR, f"Unsupported topo type: {topo_type}")
        raise ValueError(f"Unsupported output topo type: {topo_type}")

    # Create vm_params dictionary
    vm_params = {
        "model_version": model_version,
        "topo_type": TopoTypes[topo_type.upper()],
        "min_vs": min_vs,
    }

    # Ensure output directory exists
    out_dir = out_dir.resolve()
    try:
        out_dir.mkdir(exist_ok=True, parents=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {out_dir}: {e}")
        raise OSError(f"Failed to create output directory {out_dir}: {str(e)}")

    # Initialize registry
    cvm_registry = CVMRegistry(vm_params["model_version"], data_root, nzcvm_registry, logger)

    # Read profile parameters from location_csv
    try:
        df = pd.read_csv(location_csv)
        required_columns = ['id', 'lon', 'lat', 'zmin', 'zmax', 'spacing']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"CSV file {location_csv} must contain columns: {', '.join(required_columns)}")
            raise ValueError(f"Invalid CSV format: missing required columns")
        for i, row in df.iterrows():
            if row['zmin'] >= row['zmax']:
                logger.error(f"Profile {row['id']}: zmin ({row['zmin']}) must be less than zmax ({row['zmax']})")
                raise ValueError(f"Profile {row['id']}: zmin ({row['zmin']}) must be less than zmax ({row['zmax']})")
            if row['spacing'] <= 0:
                logger.error(f"Profile {row['id']}: spacing ({row['spacing']}) must be positive")
                raise ValueError(f"Profile {row['id']}: spacing ({row['spacing']}) must be positive")
    except (OSError, pd.errors.ParserError) as e:
        logger.error(f"Failed to read location_csv {location_csv}: {e}")
        raise OSError(f"Failed to read location_csv {location_csv}: {str(e)}")

    # Read custom depth points if provided
    depth_values = None
    if custom_depth:
        try:
            depth_values = read_depth_points_text_file(custom_depth, logger)
        except (OSError, ValueError) as e:
            logger.error(f"Failed to read custom depth points file {custom_depth}: {e}")
            raise

    # Load all required data
    logger.log(logging.INFO, "Loading model data")
    try:
        vm1d_data, nz_tomography_data, global_surfaces, basin_data_list = cvm_registry.load_all_global_data()
    except RuntimeError as e:
        logger.error(f"Failed to load model data: {e}")
        raise RuntimeError(f"Failed to load model data: {str(e)}")

    # Process each profile
    for i in tqdm(range(len(df)), desc="Generating profiles", unit="profile"):
        logger.log(logging.INFO, f"Generating profile {i+1} of {len(df)}")

        # Initialize model extent
        model_extent = {
            "version": vm_params["model_version"],
            "origin_rot": 0.0,
            "extent_x": 1.0,
            "extent_y": 1.0,
            "h_lat_lon": 1.0,
            "origin_lat": df['lat'].iloc[i],
            "origin_lon": df['lon'].iloc[i],
        }

        # Set depth parameters
        if depth_values:
            model_extent["extent_zmin"] = min(depth_values)
            model_extent["extent_zmax"] = max(depth_values)
            model_extent["h_depth"] = 1.0  # Placeholder, as actual depths are set later
        else:
            half = 0.5
            model_extent["extent_zmin"] = df['zmin'].iloc[i] - half * df['spacing'].iloc[i]
            model_extent["extent_zmax"] = df['zmax'].iloc[i] + half * df['spacing'].iloc[i]
            model_extent["h_depth"] = df['spacing'].iloc[i]

        model_extent["nx"] = int(model_extent["extent_x"] / model_extent["h_lat_lon"] + 0.5)
        model_extent["ny"] = int(model_extent["extent_y"] / model_extent["h_lat_lon"] + 0.5)
        model_extent["nz"] = int(
            (model_extent["extent_zmax"] - model_extent["extent_zmin"]) / model_extent["h_depth"]
            + 0.5
        )

        # Generate model grid
        try:
            global_mesh = gen_full_model_grid_great_circle(model_extent, logger)
        except RuntimeError as e:
            logger.error(f"Failed to generate model grid for profile {i+1}: {e}")
            raise RuntimeError(f"Failed to generate model grid for profile {i+1}: {str(e)}")

        # Adjust depths for custom depth points
        if depth_values:
            global_mesh.n_z = len(depth_values)
            logger.log(logging.DEBUG, f"Number of model points - nx: {global_mesh.n_x}, ny: {global_mesh.n_y}, nz: {global_mesh.n_z}")
            global_mesh.z = [-1000 * dep for dep in depth_values]

        # Initialize data structures
        try:
            partial_global_mesh = extract_partial_mesh(global_mesh, 0)
            mesh_vector = extract_mesh_vector(partial_global_mesh, 0)
            in_basin_list = [
                InBasin(basin_data, len(global_mesh.z))
                for basin_data in basin_data_list
            ]
            partial_global_surface_depths = PartialGlobalSurfaceDepths(len(global_surfaces))
            partial_basin_surface_depths = [PartialBasinSurfaceDepths(basin_data) for basin_data in basin_data_list]
            qualities_vector = QualitiesVector(partial_global_mesh.nz)
        except RuntimeError as e:
            logger.error(f"Failed to initialize data structures for profile {i+1}: {e}")
            raise RuntimeError(f"Failed to initialize data structures for profile {i+1}: {str(e)}")

        # Assign qualities
        try:
            qualities_vector.assign_qualities(
                cvm_registry,
                vm1d_data,
                nz_tomography_data,
                global_surfaces,
                basin_data_list,
                mesh_vector,
                partial_global_surface_depths,
                partial_basin_surface_depths,
                in_basin_list,
                None,  # No in_basin_mesh for profiles
                vm_params["topo_type"],
            )
        except RuntimeError as e:
            logger.error(f"Error assigning qualities for profile {i+1}: {e}")
            raise RuntimeError(f"Error assigning qualities for profile {i+1}: {str(e)}")

        # Write profile data
        try:
            write_profiles(
                out_dir,
                qualities_vector,
                vm_params,
                mesh_vector,
                df,
                i,
                logger,
            )
            write_profile_surface_depths(
                out_dir,
                cvm_registry,
                basin_data_list,
                partial_global_surface_depths,
                partial_basin_surface_depths,
                in_basin_list,
                mesh_vector,
                df,
                i,
                logger,
            )
        except OSError as e:
            logger.error(f"Error writing profile {i + 1}: {e}")
            raise OSError(f"Failed to write profile {i + 1} to {out_dir}: {str(e)}")

        logger.log(logging.INFO, f"Profile {i + 1} of {len(df)} complete")

    logger.log(logging.INFO, "Generation of multiple profiles 100% complete")
    logger.log(logging.INFO,
               f"Profiles (version: {vm_params['model_version']}) successfully generated and written to {out_dir}")
    elapsed_time = time.time() - start_time
    logger.log(logging.INFO, f"Multiple profiles generation completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    app()
