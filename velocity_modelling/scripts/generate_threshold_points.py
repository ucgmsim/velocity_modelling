"""
generate_threshold_points.py

This script generates threshold velocity values (VS30, VS500, Z1.0, Z2.5) for station locations.
It replaces the functionality of get_z.py, computing threshold values using the Python velocity model.

Default behavior (mimicking get_z.py): Computes Z1.0 and Z2.5 values with sigma.

This script is part of the velocity_modelling package and is designed to be run from the command line.

Usage:
    # Default: compute Z1.0 and Z2.5 (like get_z.py)
    python generate_threshold_points.py --station-file stations.ll --model-version 2.07

    # Compute specific threshold types
    python generate_threshold_points.py --station-file stations.ll --model-version 2.07 --vs-type VS30

    # Compute multiple threshold types
    python generate_threshold_points.py --station-file stations.ll --model-version 2.07 --vs-type Z1.0 --vs-type Z2.5 --vs-type VS30

Examples:
    # Default behavior (Z1.0, Z2.5, sigma) - output: stations.z
    python generate_threshold_points.py --station-file stations.ll --model-version 2.07

    # Custom output directory
    python generate_threshold_points.py --station-file stations.ll --model-version 2.07 --out-dir ./results

    # No header in output
    python generate_threshold_points.py --station-file stations.ll --model-version 2.07 --no-header

    # Compute VS30 only
    python generate_threshold_points.py --station-file stations.ll --model-version 2.07 --vs-type VS30

VS_TYPE options:
    - VS30: Time-averaged shear-wave velocity in the top 30 meters
    - VS500: Time-averaged shear-wave velocity in the top 500 meters
    - Z1.0: Depth (km) to Vs = 1.0 km/s
    - Z2.5: Depth (km) to Vs = 2.5 km/s

Input format:
    Station file with format: longitude latitude station_name

    Example stations.ll:
    ```
    171.747604 -43.902401 ADCS
    172.636    -43.531    CHCH
    174.768    -41.285    WTMC
    ```

Output:
    Single CSV file with computed threshold values (default: {station_file}.z)

    Example output (default: Z1.0 and Z2.5 with sigma):
    ```
    Station_Name,Z_1.0(km),Z_2.5(km),sigma
    ADCS,0.152,0.856,0.3
    CHCH,0.168,0.923,0.3
    WTMC,0.203,1.124,0.5
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
from qcore.shared import get_stations
from velocity_modelling.basin_model import (
    InBasin,
    InBasinGlobalMesh,
    PartialBasinSurfaceDepths,
)
from velocity_modelling.constants import get_data_root
from velocity_modelling.geometry import (
    MeshVector,
    PartialGlobalMesh,
    gen_full_model_grid_great_circle,
)
from velocity_modelling.global_model import PartialGlobalSurfaceDepths
from velocity_modelling.registry import CVMRegistry
from velocity_modelling.threshold import (
    VSType,
    compute_vs_average,
    compute_z_threshold,
    get_depth_parameters,
    get_z_threshold_value,
)
from velocity_modelling.velocity3d import QualitiesVector

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nzcvm")

app = typer.Typer(pretty_exceptions_enable=False)

# Constants for sigma calculation (from original get_z.py)
IN_BASIN_SIGMA = 0.3
OUT_BASIN_SIGMA = 0.5


def read_station_file(
    station_file: Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Read station file using qcore's get_stations function.

    Parameters
    ----------
    station_file : Path
        Path to the station file.
    logger : logging.Logger
        Logger instance for logging messages.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: station (index), lon, lat.

    Raises
    ------
    FileNotFoundError
        If the station file does not exist.
    ValueError
        If the file format is invalid.
    """
    if not station_file.exists():
        raise FileNotFoundError(f"Station file not found: {station_file}")

    try:
        # get_stations with locations=True returns (stations, lats, lons)
        stations, lats, lons = get_stations(str(station_file), locations=True)
    except Exception as e:
        raise ValueError(f"Failed to read station file {station_file}: {str(e)}")

    # Create DataFrame with station names as index (like get_z.py)
    df = pd.DataFrame({"lon": lons, "lat": lats}, index=stations)
    df.index.name = "Station_Name"

    logger.log(logging.INFO, f"Read {len(df)} stations from {station_file}")
    return df


def get_output_column_name(vs_type: VSType) -> str:
    """
    Get the output column name for a given threshold type.

    Parameters
    ----------
    vs_type : VSType
        Threshold type.

    Returns
    -------
    str
        Column name for the output DataFrame.
    """
    column_names = {
        VSType.VS30: "Vs_30(km/s)",
        VSType.VS500: "Vs_500(km/s)",
        VSType.Z1_0: "Z_1.0(km)",
        VSType.Z2_5: "Z_2.5(km)",
    }
    return column_names[vs_type]


@cli.from_docstring(app)
def generate_threshold_points(
    station_file: Annotated[
        Path,
        typer.Option(
            "--station-file",
            "-ll",
            exists=True,
            dir_okay=False,
            help="Station file with format: lon lat station_name",
        ),
    ],
    model_version: Annotated[
        str, typer.Option("--model-version", "-v", help="Version of the model to use")
    ] = "2.07",
    vs_type: Annotated[
        list[VSType] | None,
        typer.Option(
            "--vs-type",
            help="Threshold type(s) to compute (default: Z1.0 and Z2.5)",
        ),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            "-o",
            file_okay=False,
            help="Output directory (default: current directory)",
        ),
    ] = None,
    no_header: Annotated[
        bool,
        typer.Option(
            "--no-header",
            help="Save output with no header",
        ),
    ] = False,
    nzcvm_registry: Annotated[
        Path | None,
        typer.Option(
            exists=False,
            dir_okay=False,
            help="Path to nzcvm_registry.yaml (default: nzcvm_data/nzcvm_registry.yaml)",
        ),
    ] = None,
    nzcvm_data_root: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            exists=False,
            help="Override the default DATA_ROOT directory",
        ),
    ] = None,
    log_level: Annotated[str, typer.Option(help="Logging level")] = "INFO",
) -> None:
    """
    Generate threshold velocity values for station locations.

    Default behavior (like get_z.py): Computes Z1.0 and Z2.5 with sigma values.
    Output is saved as {station_file}.z in the specified output directory.

    This function orchestrates the computation of threshold velocity metrics:
    1. Reads station coordinates from a station file
    2. Determines which thresholds to compute (default: Z1.0, Z2.5)
    3. Loads all required velocity model data
    4. For each station location, computes velocity profile and threshold values
    5. For Z-type thresholds, calculates sigma based on basin membership
    6. Writes all results to a single CSV file

    Parameters
    ----------
    station_file : Path
        Path to station file (format: lon lat station_name).
    model_version : str, optional
        Version of the velocity model to use (default: "2.07").
    vs_type : list[VSType], optional
        Threshold type(s) to calculate. If None, defaults to [Z1.0, Z2.5].
    out_dir : Path, optional
        Output directory (default: current directory).
    no_header : bool, optional
        If True, save output without header (default: False).
    nzcvm_registry : Path, optional
        Path to the model registry file (default: nzcvm_data/nzcvm_registry.yaml).
    nzcvm_data_root : Path, optional
        Override the default nzcvm_data directory.
    log_level : str, optional
        Logging level for the script (default: "INFO").

    Raises
    ------
    FileNotFoundError
        If the station file or registry file is not found.
    ValueError
        If input parameters are invalid or threshold is outside depth limits.
    OSError
        If there are issues creating directories or writing files.
    RuntimeError
        If an error occurs during data loading or processing.
    """
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logger.setLevel(numeric_level)

    start_time = time.time()

    # Default behavior: compute Z1.0 and Z2.5 (like get_z.py)
    if vs_type is None or len(vs_type) == 0:
        vs_types = [VSType.Z1_0, VSType.Z2_5]
        logger.log(logging.INFO, "No VS_TYPE specified, defaulting to Z1.0 and Z2.5")
    else:
        vs_types = list(vs_type)

    logger.log(
        logging.INFO,
        f"Starting threshold calculation for: {', '.join([t.value for t in vs_types])}",
    )
    logger.log(logging.INFO, f"Model version: {model_version}")

    # Get data root
    data_root = get_data_root(nzcvm_data_root)
    logger.log(logging.INFO, f"Using data root: {data_root}")

    # Validate registry
    registry_path = (
        nzcvm_registry if nzcvm_registry else data_root / "nzcvm_registry.yaml"
    )
    if not registry_path.exists():
        error_msg = f"Registry file not found: {registry_path}"
        logger.log(logging.ERROR, error_msg)
        raise FileNotFoundError(error_msg)
    logger.log(logging.INFO, f"Using registry: {registry_path}")

    # Read station file
    try:
        stations_df = read_station_file(station_file, logger)
    except Exception as e:
        logger.log(logging.ERROR, f"Failed to read station file: {e}")
        raise

    # Set output directory and file
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir = out_dir.resolve()

    # Output filename: {station_file}.z
    output_file = out_dir / station_file.with_suffix(".z").name
    logger.log(logging.INFO, f"Output will be saved as {output_file}")

    # Create output directory
    try:
        out_dir.mkdir(exist_ok=True, parents=True)
    except OSError as e:
        logger.log(logging.ERROR, f"Failed to create output directory {out_dir}: {e}")
        raise OSError(f"Failed to create output directory {out_dir}: {str(e)}")

    # Initialize results DataFrame
    results_df = stations_df.copy()

    # Track basin membership for sigma calculation
    station_in_basin = {}

    # Initialize registry (only once for all threshold types)
    logger.log(logging.INFO, "Initializing velocity model registry")
    cvm_registry = CVMRegistry(model_version, data_root, nzcvm_registry, logger)

    # Load all required data once
    logger.log(logging.INFO, "Loading velocity model data")
    try:
        vm1d_data, nz_tomography_data, global_surfaces, basin_data_list = (
            cvm_registry.load_all_global_data()
        )
    except Exception as e:
        logger.log(logging.ERROR, f"Failed to load model data: {e}")
        raise RuntimeError(f"Failed to load model data: {str(e)}")

    # Process each threshold type
    for vs_type_current in vs_types:
        logger.log(logging.INFO, f"Processing {vs_type_current.value}")

        # Get depth parameters for this threshold type
        zmax, zmin, h_depth = get_depth_parameters(vs_type_current)

        # Set up model parameters
        vm_params_template = {
            "model_version": model_version,
            "origin_rot": 0,
            "extent_x": 1,
            "extent_y": 1,
            "extent_zmax": zmax,
            "extent_zmin": zmin,
            "h_depth": h_depth,
            "h_lat_lon": 1,
            "topo_type": "SQUASHED",
        }

        # Storage for threshold values for this type
        threshold_values = []

        # Process each station
        for station_name, row in tqdm(
            stations_df.iterrows(),
            total=len(stations_df),
            desc=f"Computing {vs_type_current.value}",
        ):
            lat = row["lat"]
            lon = row["lon"]

            try:
                # Create model parameters for this specific location
                vm_params = vm_params_template.copy()
                vm_params["origin_lat"] = lat
                vm_params["origin_lon"] = lon

                # Generate simple mesh for this single point
                global_mesh = gen_full_model_grid_great_circle(vm_params, None)

                # Preprocess basin membership
                in_basin_mesh = InBasinGlobalMesh(global_mesh, basin_data_list)

                # Extract mesh for the single point
                partial_global_mesh = PartialGlobalMesh(global_mesh, 0)
                mesh_vector = MeshVector(partial_global_mesh, 0)

                # Initialize data structures
                in_basin_list = [InBasin() for _ in basin_data_list]

                # Get basin indices
                basin_indices = [
                    in_basin_mesh.basin_idx[0][0][basin_idx]
                    for basin_idx, _ in enumerate(basin_data_list)
                ]

                # Mark basin membership
                in_any_basin = False
                for basin_idx in basin_indices:
                    if basin_idx >= 0:
                        in_basin_list[basin_idx].in_basin_lat_lon = True
                        in_any_basin = True

                # Track basin membership for sigma calculation (only once per station)
                if station_name not in station_in_basin:
                    station_in_basin[station_name] = in_any_basin

                # Initialize surface depths
                partial_global_surface_depths = PartialGlobalSurfaceDepths(
                    len(global_surfaces)
                )
                partial_basin_surface_depths = [
                    PartialBasinSurfaceDepths(basin_data)
                    for basin_data in basin_data_list
                ]

                # Initialize qualities vector
                qualities_vector = QualitiesVector(partial_global_mesh.nz)

                # Assign velocity/density properties
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
                    in_basin_mesh,
                    vm_params["topo_type"],
                )

                # Calculate threshold value
                if vs_type_current in [VSType.VS30, VSType.VS500]:
                    threshold_value = compute_vs_average(
                        partial_global_mesh, qualities_vector
                    )
                else:  # Z1.0 or Z2.5
                    z_thresh = get_z_threshold_value(vs_type_current)
                    threshold_value = compute_z_threshold(
                        partial_global_mesh, qualities_vector, z_thresh
                    )

                threshold_values.append(threshold_value)

            except Exception as e:
                logger.log(
                    logging.ERROR,
                    f"Error processing station {station_name} at ({lat}, {lon}): {e}",
                )
                raise RuntimeError(
                    f"Failed to process station {station_name} at ({lat}, {lon}): {str(e)}"
                )

        # Add threshold values to results DataFrame
        column_name = get_output_column_name(vs_type_current)
        results_df[column_name] = threshold_values

    # Calculate sigma if any Z-type thresholds were computed
    if any(vt in [VSType.Z1_0, VSType.Z2_5] for vt in vs_types):
        logger.log(logging.INFO, "Calculating sigma values based on basin membership")
        # Set sigma based on whether station is in any basin
        # IN_BASIN_SIGMA (0.3) if in basin, OUT_BASIN_SIGMA (0.5) if not
        sigma_values = []
        n_in_basin = 0
        n_out_basin = 0
        for station_name in results_df.index:
            if station_in_basin.get(station_name, False):
                sigma_values.append(IN_BASIN_SIGMA)
                n_in_basin += 1
            else:
                sigma_values.append(OUT_BASIN_SIGMA)
                n_out_basin += 1
        results_df["sigma"] = sigma_values
        logger.log(
            logging.INFO,
            f"Sigma calculation complete: {n_in_basin} stations in basin (σ={IN_BASIN_SIGMA}), "
            f"{n_out_basin} stations outside basin (σ={OUT_BASIN_SIGMA})",
        )

    # Write results to file
    logger.log(logging.INFO, f"Writing results to {output_file}")
    try:
        results_df.to_csv(output_file, header=not no_header)
        logger.log(logging.INFO, f"Results successfully written to {output_file}")
    except Exception as e:
        logger.log(logging.ERROR, f"Failed to write results: {e}")
        raise OSError(f"Failed to write results to {output_file}: {str(e)}")

    # Print preview (like get_z.py)
    print(results_df)

    logger.log(logging.INFO, "Threshold calculation 100% complete")
    elapsed_time = time.time() - start_time
    logger.log(
        logging.INFO,
        f"Threshold calculation completed in {elapsed_time:.2f} seconds",
    )


if __name__ == "__main__":
    app()