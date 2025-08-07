"""
generate_1d_profiles.py

This script generates multiple 1D velocity profiles for specified geographic locations and depth intervals.
It reads profile parameters from a CSV file, optionally accepts custom depth points, loads model data,
computes velocity and density profiles, and writes results to disk in standard or site response formats.
Intended for use with the NZCVM velocity modelling framework.

This script is part of the velocity_modelling package and is designed to be run from the command line.
Usage:
    python generate_1d_profiles.py --out-dir <output_directory> --model-version <version> --location-csv <csv_file> --min-vs <min_vs> --topo-type <topo_type> [--custom-depth <depth_file>] [--nzcvm-registry <registry_file>] [--data-root <data_root>] [--log-level <log_level>]

Example:
    python generate_1d_profiles.py --out-dir ./profiles --model-version 2.07 --location-csv locations.csv --min-vs 0.2 --topo-type TRUE

    where, locations.csv is a CSV file with columns: id, lon, lat, zmin, zmax, spacing. zmin, zmax and spacing are in kilometers.

    Sample locations.csv:
        id, lon, lat, zmin, zmax, spacing
        ADCS, 171.747604, -43.902401,0.0, 3.0, 0.1
        ...
If the --custom-depth option is provided, it should point to a text file with depth points in kilometers, one per line, such as:
        0.0
        0.1
        0.5
        1.0
        5.0
In this case, the zmin, zmax, and spacing parameters in the CSV file will be ignored.


Sample output:  Profile_ADCS.txt
```
Properties at Lat : -43.902401 Lon: 171.747604 (On Mesh Lat: -43.902396 Lon: 171.747599)
Model Version: 2.07
Topo Type: TRUE
Minimum Vs: 0.000000
Elevation (km) 	 Vp (km/s) 	 Vs (km/s) 	 Rho (g/cm^3)
-0.000000 	 1.800000 	 0.380000 	 1.810000
-0.100000 	 1.800000 	 0.480000 	 1.810000
-0.200000 	 1.800000 	 0.608600 	 1.810000
-0.300000 	 2.000000 	 0.608600 	 1.905000
...

-3.000000 	 4.520129 	 2.676067 	 2.538085
```
Sample output: ProfileSurfaceDepths_ADCS.txt
```
Surface Elevation (in m) at Lat : -43.902401 Lon: 171.747604 (On Mesh Lat: -43.902396 Lon: 171.747599)

Global surfaces
Surface_name 	 Elevation (m)
- posInf	1000000.000000
- NZ_DEM_HD	99.607015
- negInf	-1000000.000000

Basin surfaces (if applicable)

Canterbury_v19p1
- CantDEM	99.607015
- Canterbury_Pliocene_46_WGS84_v8p9p18	-266.438467
- Canterbury_Miocene_WGS84	-887.287466
- Canterbury_Paleogene_WGS84	-1031.612588
- Canterbury_basement_WGS84	-1765.060557

```

"""

import logging
import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from qcore import cli
from velocity_modelling.basin_model import (
    BasinData,
    InBasin,
    InBasinGlobalMesh,
    PartialBasinSurfaceDepths,
)
from velocity_modelling.constants import DATA_ROOT, NZCVM_REGISTRY_PATH, TopoTypes
from velocity_modelling.geometry import (
    MeshVector,
    gen_full_model_grid_great_circle,
)
from velocity_modelling.global_model import (
    GlobalSurfaceRead,
    PartialGlobalSurfaceDepths,
)
from velocity_modelling.registry import CVMRegistry
from velocity_modelling.velocity3d import QualitiesVector

LAST_LAYER_DEPTH = -999999.0

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

    The file is expected to contain one depth value (in kilometres) per line.

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
    if not file_path.exists():
        logger.error(f"Depth points file does not exist: {file_path}")
        raise OSError(f"Depth points file does not exist: {file_path}")
    try:
        # Use numpy to read the file and convert to a list.
        # atleast_1d ensures we get a 1D array even if the file has a single value.
        depth_values = np.atleast_1d(np.loadtxt(file_path, dtype=float)).tolist()
    except (OSError, ValueError) as e:
        logger.error(f"Failed to read depth points file {file_path}: {e}")
        raise
    if not depth_values:
        logger.log(logging.ERROR, f"No valid depth points found in {file_path}")
        raise ValueError(f"No valid depth points found in {file_path}")
    logger.log(logging.DEBUG, f"Read {len(depth_values)} depth points from {file_path}")
    return depth_values


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
    profiles_dir.mkdir(exist_ok=True, parents=True)
    output_type = vm_params.get(
        "output_type", "STANDARD"
    )  # Default to STANDARD if not specified
    profile_id = df["id"].iloc[profile_idx]

    if output_type == "1D_SITE_RESPONSE":
        file_path = profiles_dir / f"{profile_id}.1d"
        with file_path.open("w") as f:
            f.write(f"{mesh_vector.nz}\n")
            dep_bot = 0.0
            for i in range(mesh_vector.nz):
                vs = max(qualities_vector.vs[i], vm_params["min_vs"])
                if i == mesh_vector.nz - 1:
                    delta_depth = LAST_LAYER_DEPTH
                elif i == 0:
                    delta_depth = 2 * mesh_vector.z[i]
                    dep_bot = delta_depth
                else:
                    delta_depth = 2 * (mesh_vector.z[i] - dep_bot)
                    dep_bot += delta_depth
                quality_factor_1 = 2.0 * (41.0 + 34.0 * vs)
                quality_factor_2 = 41.0 + 34.0 * vs
                f.write(
                    f"{-delta_depth / 1000:.3f} \t {qualities_vector.vp[i]:.3f} \t "
                    f"{vs:.3f} \t {qualities_vector.rho[i]:.3f} \t "
                    f"{quality_factor_1:.3f} \t {quality_factor_2:.3f}\n"
                )
        logger.log(logging.INFO, f"Wrote 1D site response profile to {file_path}")

    elif output_type == "STANDARD":
        file_path = profiles_dir / f"Profile_{profile_id}.txt"
        with file_path.open("w") as f:
            f.write(
                f"Properties at Lat : {df['lat'].iloc[profile_idx]:.6f} Lon: {df['lon'].iloc[profile_idx]:.6f} (On Mesh Lat: {mesh_vector.lat:.6f} Lon: {mesh_vector.lon:.6f})\n"
            )
            f.write(f"Model Version: {vm_params['model_version']}\n")
            f.write(f"Topo Type: {vm_params['topo_type'].name}\n")
            f.write(f"Minimum Vs: {vm_params['min_vs']:.6f}\n")
            f.write("Elevation (km) \t Vp (km/s) \t Vs (km/s) \t Rho (g/cm^3)\n")
            for i in range(mesh_vector.nz):
                vs = max(qualities_vector.vs[i], vm_params["min_vs"])
                f.write(
                    f"{mesh_vector.z[i] / 1000:.6f} \t {qualities_vector.vp[i]:.6f} \t "
                    f"{vs:.6f} \t {qualities_vector.rho[i]:.6f}\n"
                )
        logger.log(logging.INFO, f"Wrote standard profile to {file_path}")

    else:
        logger.log(logging.ERROR, f"Invalid output_type: {output_type}")
        raise ValueError(
            f"Invalid output_type: {output_type}. Must be '1D_SITE_RESPONSE' or 'STANDARD'"
        )


def write_profile_surface_depths(
    out_dir: Path,
    global_surfaces: list[GlobalSurfaceRead],
    basin_data_list: list[BasinData],
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
    global_surfaces : list[GlobalSurfaceRead]
        List of global surface data objects.
    basin_data_list : list[BasinData]
        List of basin data objects.
    partial_global_surface_depths : PartialGlobalSurfaceDepths
        Depths of global surfaces at the profile location.
    partial_basin_surface_depths : PartialBasinSurfaceDepths
        Depths of basin surfaces at the profile location.
    in_basin_list : list[InBasin]
        List of InBasin objects indicating basin membership for the profile location.
    mesh_vector : QualitiesVector
        Mesh vector containing latitude and longitude of the profile location.
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
    profiles_dir.mkdir(exist_ok=True, parents=True)
    file_path = profiles_dir / f"ProfileSurfaceDepths_{df['id'].iloc[profile_idx]}.txt"
    lines = [
        f"Surface Elevation (in m) at Lat : {df['lat'].iloc[profile_idx]:.6f} Lon: {df['lon'].iloc[profile_idx]:.6f} (On Mesh Lat: {mesh_vector.lat:.6f} Lon: {mesh_vector.lon:.6f})\n",
        "\nGlobal surfaces\n",
        "Surface_name \t Elevation (m)\n",
        *[
            f"- {Path(global_surfaces[i].file_path).stem}\t{partial_global_surface_depths.depths[i]:.6f}\n"
            for i in range(len(global_surfaces))
        ],
        "\nBasin surfaces (if applicable)\n",
    ]

    for i, basin in enumerate(basin_data_list):
        if in_basin_list[i].in_basin_lat_lon:
            lines.append(f"\n{basin.name}\n")
            lines.extend(
                f"- {Path(basin.surfaces[j].file_path).stem}\t{partial_basin_surface_depths[i].depths[j]:.6f}\n"
                for j in range(len(basin.surfaces))
            )

    with file_path.open("w") as f:
        f.writelines(lines)
    logger.log(logging.INFO, f"Wrote surface depths to {file_path}")


@cli.from_docstring(app)
def generate_1d_profiles(
    out_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Output directory for profile files")
    ],
    model_version: Annotated[str, typer.Option(help="Version of the model to use")],
    location_csv: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="CSV file with profile parameters (id, lon, lat, zmin, zmax, spacing)",
        ),
    ],
    min_vs: Annotated[float, typer.Option(help="Minimum shear wave velocity")] = 0.0,
    topo_type: Annotated[
        str, typer.Option(help="Topography type")
    ] = TopoTypes.TRUE.name,
    custom_depth: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Text file with custom depth points (overrides zmin, zmax, spacing in location_csv)",
        ),
    ] = None,
    nzcvm_registry: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = NZCVM_REGISTRY_PATH,
    data_root: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            exists=True,
            help="Override the default DATA_ROOT directory",
        ),
    ] = DATA_ROOT,
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
        logger.log(logging.ERROR, f"min_vs ({min_vs}) cannot be negative")
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
    out_dir.mkdir(exist_ok=True, parents=True)
    cvm_registry = CVMRegistry(
        vm_params["model_version"], data_root, nzcvm_registry, logger
    )

    # Read profile parameters from location_csv
    try:
        df = pd.read_csv(location_csv, skipinitialspace=True)
    except pd.errors.EmptyDataError:
        logger.log(logging.ERROR, f"CSV file {location_csv} is empty or invalid.")
        raise ValueError(f"CSV file {location_csv} is empty or invalid.")

    # Standardize column names to lowercase to be forgiving
    df.columns = df.columns.str.lower()

    required_columns = ["id", "lon", "lat", "zmin", "zmax", "spacing"]
    if list(df.columns) != required_columns:
        # Check if it looks like there's no header vs. a wrong header
        if not any(col in df.columns for col in required_columns):
            message = f"CSV file {location_csv} appears to be missing a header."
        else:
            message = f"CSV file {location_csv} has an incorrect header."
        logger.log(
            logging.ERROR,
            f"{message} Expected columns in order: {', '.join(required_columns)}",
        )
        raise ValueError("Invalid CSV format: incorrect or missing header")

    for i, row in df.iterrows():
        if row["zmin"] >= row["zmax"]:
            logger.log(
                logging.ERROR,
                f"Profile {row['id']}: zmin ({row['zmin']}) must be less than zmax ({row['zmax']})",
            )
            raise ValueError(
                f"Profile {row['id']}: zmin ({row['zmin']}) must be less than zmax ({row['zmax']})"
            )
        if row["spacing"] <= 0:
            logger.log(
                logging.ERROR,
                f"Profile {row['id']}: spacing ({row['spacing']}) must be positive",
            )
            raise ValueError(
                f"Profile {row['id']}: spacing ({row['spacing']}) must be positive"
            )
    depth_values = None
    if custom_depth:
        depth_values = read_depth_points_text_file(custom_depth, logger)
    logger.log(logging.INFO, "Loading model data")
    vm1d_data, nz_tomography_data, global_surfaces, basin_data_list = (
        cvm_registry.load_all_global_data()
    )
    for i in tqdm(range(len(df)), desc="Generating profiles", unit="profile"):
        logger.log(logging.INFO, f"Generating profile {i + 1} of {len(df)}")

        # Initialize model extent
        model_extent = {
            "version": vm_params["model_version"],
            "origin_rot": 0.0,
            "extent_x": 1.0,
            "extent_y": 1.0,
            "h_lat_lon": 1.0,
            "origin_lat": df["lat"].iloc[i],
            "origin_lon": df["lon"].iloc[i],
        }

        # Set depth parameters
        if depth_values:
            model_extent["extent_zmin"] = min(depth_values)
            model_extent["extent_zmax"] = max(depth_values)
            model_extent["h_depth"] = 1.0  # Placeholder, as actual depths are set later
        else:
            spacing_offset = 0.5
            model_extent["extent_zmin"] = (
                df["zmin"].iloc[i] - spacing_offset * df["spacing"].iloc[i]
            )
            model_extent["extent_zmax"] = (
                df["zmax"].iloc[i] + spacing_offset * df["spacing"].iloc[i]
            )
            model_extent["h_depth"] = df["spacing"].iloc[i]
        model_extent["nx"] = int(
            model_extent["extent_x"] / model_extent["h_lat_lon"] + 0.5
        )
        model_extent["ny"] = int(
            model_extent["extent_y"] / model_extent["h_lat_lon"] + 0.5
        )
        model_extent["nz"] = int(
            (model_extent["extent_zmax"] - model_extent["extent_zmin"])
            / model_extent["h_depth"]
            + 0.5
        )
        global_mesh = gen_full_model_grid_great_circle(model_extent, logger)
        logger.log(logging.INFO, "Pre-processing basin membership")
        in_basin_mesh, partial_global_mesh_list = (
            InBasinGlobalMesh.preprocess_basin_membership(
                global_mesh,
                basin_data_list,
                logger,
                smooth_bound=nz_tomography_data.smooth_boundary,
            )
        )
        if depth_values:
            global_mesh.nz = len(depth_values)
            logger.log(
                logging.DEBUG,
                f"Number of model points - nx: {global_mesh.nx}, ny: {global_mesh.ny}, nz: {global_mesh.nz}",
            )
            global_mesh.z = np.array([-1000 * dep for dep in depth_values])  # in meters

        partial_global_mesh = partial_global_mesh_list[
            0
        ]  # Use the first mesh for the profile
        basin_indices = in_basin_mesh.get_basin_membership(
            0, 0
        )  # Get basin indices for the profile location
        in_basin_list = [
            InBasin(basin_data, len(global_mesh.z)) for basin_data in basin_data_list
        ]
        for basin_idx in basin_indices:
            if basin_idx >= 0:
                in_basin_list[basin_idx].in_basin_lat_lon = True
        mesh_vector = MeshVector(partial_global_mesh, 0)
        partial_global_surface_depths = PartialGlobalSurfaceDepths(len(global_surfaces))
        partial_basin_surface_depths = [
            PartialBasinSurfaceDepths(basin_data) for basin_data in basin_data_list
        ]
        qualities_vector = QualitiesVector(partial_global_mesh.nz)
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
            None,
            vm_params["topo_type"],
        )
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
            global_surfaces,
            basin_data_list,
            partial_global_surface_depths,
            partial_basin_surface_depths,
            in_basin_list,
            mesh_vector,
            df,
            i,
            logger,
        )
        logger.log(logging.INFO, f"Profile {i + 1} of {len(df)} complete")
    logger.log(logging.INFO, "Generation of multiple profiles 100% complete")
    logger.log(
        logging.INFO,
        f"Profiles (version: {vm_params['model_version']}) successfully generated and written to {out_dir}",
    )
    elapsed_time = time.time() - start_time
    logger.log(
        logging.INFO,
        f"Multiple profiles generation completed in {elapsed_time:.2f} seconds",
    )


if __name__ == "__main__":
    app()
