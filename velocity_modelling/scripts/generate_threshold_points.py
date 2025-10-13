"""
generate_threshold_points.py

This script computes threshold velocity/depth values (e.g., Vs30, Vs500, Z1.0, Z2.5)
for station locations using the NZCVM Python implementation. It reads station
coordinates, determines basin membership, computes requested thresholds, and writes
results to CSV.

Notes
-----
- Output CSV includes Station_Name as index and computed threshold columns (and sigma if applicable).
  Longitude and latitude columns are not included in the output.
- Default behavior computes Z1.0 and Z2.5 (like get_z.py).
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
    PartialBasinSurfaceDepths,
    StationBasinMembership,
    compute_sigma_for_stations,
)
from velocity_modelling.constants import TopoTypes, get_data_root
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


def read_station_file(station_file: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Read a station file and return station coordinates.

    Parameters
    ----------
    station_file : Path
        Path to the station file (format: lon lat station_name).
    logger : logging.Logger
        Logger instance for reporting errors.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Station_Name with columns:
        - lon: float, station longitude
        - lat: float, station latitude

    Raises
    ------
    FileNotFoundError
        If the station file does not exist.
    ValueError
        If the station file cannot be parsed.
    """
    if not station_file.exists():
        raise FileNotFoundError(f"Station file not found: {station_file}")

    try:
        stations, lats, lons = get_stations(str(station_file), locations=True)
    except (OSError, ValueError) as e:
        raise ValueError(f"Failed to read station file {station_file}: {str(e)}")

    df = pd.DataFrame({"lon": lons, "lat": lats}, index=stations)
    df.index.name = "Station_Name"
    return df


def get_output_column_name(vs_type: VSType) -> str:
    """
    Map a VSType to its output column name.

    Parameters
    ----------
    vs_type : VSType
        The threshold type to compute.

    Returns
    -------
    str
        Output column header corresponding to the threshold type.
    """
    if vs_type in [VSType.VS30, VSType.VS500]:
        depth_meters = "30" if vs_type == VSType.VS30 else "500"
        return f"Vs_{depth_meters}(km/s)"
    else:
        return f"Z_{vs_type.value}(km)"


@cli.from_docstring(app)
def generate_threshold_points(
    station_file: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
        ),
    ],
    model_version: str = "2.07",
    vs_type: list[VSType] | None = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
        ),
    ] = None,
    no_header: bool = False,
    nzcvm_registry: Annotated[
        Path | None,
        typer.Option(
            exists=False,
            dir_okay=False,
        ),
    ] = None,
    nzcvm_data_root: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            exists=False,
        ),
    ] = None,
    log_level: str = "INFO",
) -> None:
    """
    Generate threshold values (Vs30, Vs500, Z1.0, Z2.5) for station locations.

    This computes requested thresholds for each station, using precomputed basin
    membership for sigma assignment when Z-thresholds are requested.

    Parameters
    ----------
    station_file : Path
        Station file path (format: lon lat station_name).
    model_version : str
        NZCVM model version to use (default: "2.07").
    vs_type : list[VSType] | None
        Threshold types to compute. If None, computes [Z1.0, Z2.5].
    out_dir : Path | None
        Output directory (default: current working directory).
    no_header : bool
        If True, write CSV without header row.
    nzcvm_registry : Path | None
        Optional path to nzcvm_registry.yaml; defaults to registry under data root.
    nzcvm_data_root : Path | None
        Override for data root directory; if None, use configured default.
    log_level : str
        Logging level, e.g., "INFO", "DEBUG".

    Returns
    -------
    None
        Writes results to CSV named {station_file}.csv in the output directory.

    Raises
    ------
    FileNotFoundError
        If registry or station file cannot be found.
    ValueError
        For invalid inputs or configuration issues.
    RuntimeError
        If model data loading or per-station processing fails.
    OSError
        If output directory cannot be created or output file cannot be written.
    """
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logger.setLevel(numeric_level)

    start_time = time.time()

    # Default behavior: compute Z1.0 and Z2.5 (like get_z.py)
    if not vs_type: # if vs_type is None or []
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

    logger.log(logging.INFO, f"Read {len(stations_df)} stations from {station_file}")

    # Set output directory and file
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir = out_dir.resolve()

    output_file = out_dir / station_file.with_suffix(".csv").name
    logger.log(logging.INFO, f"Output will be saved as {output_file}")

    # Create output directory
    try:
        out_dir.mkdir(exist_ok=True, parents=True)
    except OSError as e:
        logger.log(logging.ERROR, f"Failed to create output directory {out_dir}: {e}")
        raise OSError(f"Failed to create output directory {out_dir}: {str(e)}")

    # Initialize results DataFrame
    results_df = stations_df.copy()

    # Initialize registry (only once for all threshold types)
    logger.log(logging.INFO, "Initializing velocity model registry")
    cvm_registry = CVMRegistry(model_version, data_root, nzcvm_registry, logger)

    # Load all required data once
    logger.log(logging.INFO, "Loading velocity model data")
    try:
        vm1d_data, nz_tomography_data, global_surfaces, basin_data_list = (
            cvm_registry.load_all_global_data()
        )
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        logger.log(logging.ERROR, f"Failed to load model data: {e}")
        raise RuntimeError(f"Failed to load model data: {str(e)}")

    # Basin membership pre-processing
    logger.log(logging.INFO, "Pre-computing basin membership for all stations")

    # Initialize station basin membership checker (no mesh needed!)
    station_basin_checker = StationBasinMembership(basin_data_list, logger)

    # Check basin membership for ALL stations at once
    station_lats = stations_df["lat"].values
    station_lons = stations_df["lon"].values
    station_basin_membership = station_basin_checker.check_stations_in_basin(
        station_lats, station_lons
    )

    logger.log(
        logging.INFO, f"Basin membership computed for {len(stations_df)} stations"
    )

    # Compute sigma values once (for Z-type thresholds)
    sigma_values = None
    if any(vt in [VSType.Z1_0, VSType.Z2_5] for vt in vs_types):
        sigma_values = compute_sigma_for_stations(
            station_basin_membership, IN_BASIN_SIGMA, OUT_BASIN_SIGMA
        )
        n_in_basin = np.sum(sigma_values == IN_BASIN_SIGMA)
        n_out_basin = np.sum(sigma_values == OUT_BASIN_SIGMA)
        logger.log(
            logging.INFO,
            f"Sigma values computed: {n_in_basin} stations in basin (σ={IN_BASIN_SIGMA}), "
            f"{n_out_basin} stations outside basin (σ={OUT_BASIN_SIGMA})",
        )

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
            "topo_type": TopoTypes.SQUASHED,
        }

        nx = 1
        ny = 1
        nz = int((zmax - zmin) / h_depth + 0.5)

        vm_params_template["nx"] = nx
        vm_params_template["ny"] = ny
        vm_params_template["nz"] = nz

        # Storage for threshold values for this type
        threshold_values = []

        # Process each station (INNER LOOP)
        for idx, (station_name, row) in enumerate(
            tqdm(
                stations_df.iterrows(),
                total=len(stations_df),
                desc=f"Computing {vs_type_current.value}",
            )
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

                # Use pre-computed basin membership
                basin_indices = station_basin_membership[idx]

                # Create simple InBasin objects without mesh preprocessing
                in_basin_list = [
                    InBasin(basin_data, len(global_mesh.z))
                    for basin_data in basin_data_list
                ]

                # Mark basin membership using pre-computed results
                for basin_id in basin_indices:
                    if basin_id >= 0:
                        in_basin_list[basin_id].in_basin_lat_lon = True

                # Create a minimal PartialGlobalMesh for this point
                partial_global_mesh = PartialGlobalMesh(global_mesh, 0)
                mesh_vector = MeshVector(partial_global_mesh, 0)

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
                    None,  # No MeshBasinMembership for isolated station processing
                    vm_params["topo_type"],
                )

                # Calculate threshold value
                if vs_type_current in [VSType.VS30, VSType.VS500]:
                    threshold_value = compute_vs_average(
                        partial_global_mesh, qualities_vector
                    )
                else:  # Z1.0 or Z2.5
                    if vs_type_current == VSType.Z1_0:
                        z_thresh = 1.0
                    elif vs_type_current == VSType.Z2_5:
                        z_thresh = 2.5
                    else:
                        raise ValueError(
                            f"VS_TYPE {vs_type_current} is not a Z-type threshold"
                        )

                    threshold_value = compute_z_threshold(
                        partial_global_mesh, qualities_vector, z_thresh
                    )

                threshold_values.append(threshold_value)

            except (ValueError, KeyError, RuntimeError, OSError) as e:
                logger.log(
                    logging.ERROR,
                    f"Error processing station {station_name} at ({lat}, {lon}): {e}",
                )
                raise RuntimeError(
                    f"Failed to process station {station_name} at ({lat}, {lon}): {str(e)}"
                )

        print("")  # Newline after tqdm progress bar

        # Add threshold values to results DataFrame
        column_name = get_output_column_name(vs_type_current)
        results_df[column_name] = threshold_values

        logger.log(
            logging.INFO,
            f"Completed {vs_type_current.value} calculation for {len(stations_df)} stations",
        )

    # Add pre-computed sigma values if computed
    if sigma_values is not None:
        results_df["sigma"] = sigma_values
        logger.log(logging.INFO, "Added pre-computed sigma values to results")

    # Reorder columns: only threshold values (and sigma), no lon/lat in output
    output_columns = [get_output_column_name(vt) for vt in vs_types]
    if sigma_values is not None:
        output_columns.append("sigma")
    results_df = results_df[output_columns]

    # Write results to file
    logger.log(logging.INFO, f"Writing results to {output_file}")
    try:
        results_df.to_csv(
            output_file,
            index=True,
            index_label="Station_Name",
            header=not no_header,
            float_format="%.3f",
        )
        logger.log(logging.INFO, f"Results successfully written to {output_file}")
    except (OSError, ValueError) as e:
        logger.log(logging.ERROR, f"Failed to write results: {e}")
        raise OSError(f"Failed to write results to {output_file}: {str(e)}")

    logger.log(logging.INFO, "Threshold calculation 100% complete")
    elapsed_time = time.time() - start_time
    logger.log(
        logging.INFO,
        f"Threshold calculation completed in {elapsed_time:.2f} seconds",
    )


if __name__ == "__main__":
    app()
