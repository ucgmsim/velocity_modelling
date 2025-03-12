"""
Velocity Model Generator

This script generates a 3D seismic velocity model based on input parameters.
It creates a mesh grid representation of subsurface properties (P-wave velocity,
S-wave velocity, and density) by combining global velocity models, regional tomographic
models, and local basin models.

# Installation
cd velocity_modelling
pip install -e  .

# Execution

    # Basic usage : to use everything defined in nzvm.cfg
    nzvm generate-velocity-model  /path/to/nzvm.cfg

    # To override output directory
    nzvm generate-velocity-model /path/to/nzvm.cfg --out-dir /path/to/output_dir

    # With custom registry location:
    nzvm generate-velocity-model /path/to/nzvm.cfg  --nzvm-registry /path/to/registry.yaml

    # To override "MODEL_VERSION" in nzvm.cfg to use a .yaml file for a custom model version
    nzvm generate-velocity-model /path/to/nzvm.cfg  --model-version 2.07

    # With specific log level:
    nzvm generate-velocity-model /path/to/nzvm.cfg  --log-level DEBUG

    # With specific output format:
    nzvm generate-velocity-model /path/to/nzvm.cfg  --output-format CSV  (default: emod3d)


    [example] nzvm.cfg

    CALL_TYPE=GENERATE_VELOCITY_MOD
    MODEL_VERSION=2.07
    ORIGIN_LAT=-41.296226
    ORIGIN_LON=174.774439
    ORIGIN_ROT=23.0
    EXTENT_X=20
    EXTENT_Y=20
    EXTENT_ZMAX=45.0
    EXTENT_ZMIN=0.0
    EXTENT_Z_SPACING=0.2
    EXTENT_LATLON_SPACING=0.2
    MIN_VS=0.5
    TOPO_TYPE=BULLDOZED
    OUTPUT_DIR=/tmp
"""

# import concurrent.futures

import time
from pathlib import Path
from typing import Annotated

import typer

from qcore import cli
from velocity_modelling.cvm.basin_model import (
    InBasin,
    InBasinGlobalMesh,
    PartialBasinSurfaceDepths,
)
from velocity_modelling.cvm.constants import (
    NZVM_REGISTRY_PATH,
    TopoTypes,
    WriteFormat,
)
from velocity_modelling.cvm.geometry import (
    GlobalMesh,
    extract_mesh_vector,
    gen_full_model_grid_great_circle,
)
from velocity_modelling.cvm.global_model import (
    PartialGlobalSurfaceDepths,
)
from velocity_modelling.cvm.logging import VMLogger
from velocity_modelling.cvm.registry import CVMRegistry
from velocity_modelling.cvm.velocity3d import (
    PartialGlobalQualities,
    QualitiesVector,
)

app = typer.Typer(pretty_exceptions_enable=False)


def write_velo_mod_corners_text_file(
    global_mesh: GlobalMesh, output_dir: Path | str, logger: VMLogger
) -> None:
    """
    Write velocity model corners to a text file for reference and visualization.

    Records the geographic coordinates of model corners (min/max latitude and longitude)
    for later reference and plotting.

    Parameters
    ----------
    global_mesh : GlobalMesh
        An object containing the global mesh data with longitude and latitude arrays.
    output_dir : Path or str
        Directory where the log file will be saved.
    logger : VMLogger
        Logger for reporting progress and errors.
    """
    log_file_name = Path(output_dir) / "Log" / "VeloModCorners.txt"
    log_file_name.parent.mkdir(parents=True, exist_ok=True)

    nx = len(global_mesh.x)
    ny = len(global_mesh.y)

    logger.log(f"Writing velocity model corners to {log_file_name}", logger.INFO)
    try:
        with log_file_name.open("w") as fp:
            fp.write(">Velocity model corners.\n")
            fp.write(">Lon\tLat\n")
            fp.write(f"{global_mesh.lon[0][ny - 1]}\t{global_mesh.lat[0][ny - 1]}\n")
            fp.write(f"{global_mesh.lon[0][0]}\t{global_mesh.lat[0][0]}\n")
            fp.write(f"{global_mesh.lon[nx - 1][0]}\t{global_mesh.lat[nx - 1][0]}\n")
            fp.write(
                f"{global_mesh.lon[nx - 1][ny - 1]}\t{global_mesh.lat[nx - 1][ny - 1]}\n"
            )
        logger.log("Velocity model corners file write complete.", logger.INFO)
    except Exception as e:
        logger.log(f"Failed to write velocity model corners: {e}", logger.ERROR)
        raise


@cli.from_docstring(app)
def generate_velocity_model(
    nzvm_cfg_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out_dir: Annotated[Path, typer.Option(file_okay=False)],
    nzvm_registry: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = NZVM_REGISTRY_PATH,
    model_version: Annotated[str, typer.Option()] = None,
    output_format: Annotated[str, typer.Option()] = WriteFormat.EMOD3D.name,
    smoothing: Annotated[bool, typer.Option()] = False,
    progress_interval: Annotated[int, typer.Option()] = 5,
) -> None:
    """
    Generate a 3D velocity model and write it to disk.

    This is the main function that orchestrates the velocity model generation process:
    1. Creates the model mesh grid
    2. Loads all required datasets (global models, tomography, basins)
    3. Processes each latitude slice and populates velocity/density values
    4. Writes results to disk in EMOD3D format

    Parameters
    ----------
    nzvm_cfg_path : Path
        Path to the nzvm.cfg configuration file.
    out_dir : Path, Optional
        Path to the output directory where the velocity model files will be written.
         If not provided, the output directory specified in nzvm.cfg will be used.
    nzvm_registry : Path, optional
        Path to the model registry file (default: NZVM_REGISTRY_PATH).
    model_version : str, optional
        Version of the model to use (overrides MODEL_VERSION in config file).
    output_format : str, optional
        Format to write the output. Options: "EMOD3D", "CSV"
    smoothing : bool, optional
        Unsupported option for future smoothing implementation at model boundaries (default False).
    progress_interval : int, optional
        How often (in %) to log progress updates (default 5%).

    """
    start_time = time.time()

    # Import the appropriate writer based on format
    # Configure logging
    log_level = VMLogger.INFO
    logger = VMLogger(level=log_level)
    logger.log(f"Logger initialized with level {log_level}", logger.DEBUG)
    logger.log(f"Beginning velocity model generation in {out_dir}", logger.INFO)

    # Parse the config file
    try:
        vm_params = parse_nzvm_config(nzvm_cfg_path)
        # Validate CALL_TYPE
        if vm_params.get("call_type") != "GENERATE_VELOCITY_MOD":
            logger.log(
                f"Unsupported CALL_TYPE: {vm_params.get('call_type')}", logger.ERROR
            )
            sys.exit(1)
    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            raise  # Re-raise these critical exceptions
        logger.log(f"Failed to parse config file: {e}", logger.ERROR)
        sys.exit(1)

    # Use --model-version if provided, otherwise fall back to MODEL_VERSION from config
    if model_version and model_version != vm_params.get("model_version"):
        logger.log(
            f"UPDATING model_version fom {vm_params['model_version']} to {model_version}",
            logger.INFO,
        )
        vm_params["model_version"] = model_version  # Ensure version is set in vm_params
    else:
        logger.log(f"Using model version: {vm_params['model_version']}", logger.INFO)

    # Import the appropriate writer based on format
    try:
        _ = WriteFormat[output_format.upper()]
    except KeyError:
        logger.log(f"Unsupported output format: {output_format}", logger.ERROR)
        raise ValueError(f"Unsupported output format: {output_format}")
    logger.log(f"Using output format: {output_format}", logger.INFO)

    import importlib

    try:
        module_name = f"velocity_modelling.cvm.write.{output_format.lower()}"
        writer_module = importlib.import_module(module_name)
        write_global_qualities = writer_module.write_global_qualities
        logger.log(f"Using {output_format} writer module", logger.DEBUG)
    except ImportError:
        logger.log(f"Unsupported output format: {output_format}", logger.ERROR)
        raise ValueError(f"Unsupported output format: {output_format}")

    out_dir = out_dir.resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    # Initialize registry and generate model
    cvm_registry = CVMRegistry(
        vm_params["model_version"],
        logger,
        nzvm_registry,
    )
    elapsed_time = time.time() - start_time
    logger.log(
        f"Velocity model generation completed in {elapsed_time:.2f} seconds",
        logger.INFO,
    )

    # Create model grid
    logger.log("Generating model grid", logger.INFO)

    global_mesh = gen_full_model_grid_great_circle(vm_params, logger)

    write_velo_mod_corners_text_file(global_mesh, out_dir, logger)

    # Load all required data
    logger.log("Loading model data", logger.INFO)
    velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data_list = (
        cvm_registry.load_all_global_data()
    )

    # Preprocess basin membership for efficiency
    logger.log("Pre-processing basin membership", logger.INFO)
    in_basin_mesh, partial_global_mesh_list = (
        InBasinGlobalMesh.preprocess_basin_membership(
            global_mesh,
            basin_data_list,
            logger,
            smooth_bound=nz_tomography_data.smooth_boundary,
        )
    )

    # Process each latitude slice
    total_slices = len(global_mesh.y)
    last_progress = -progress_interval
    for j in range(total_slices):
        # Log progress at specified intervals
        progress = j * 100 / total_slices
        if progress >= last_progress + progress_interval:
            logger.log(
                f"Generating velocity model: {progress:.2f}% complete", logger.INFO
            )
            last_progress = progress

        partial_global_mesh = partial_global_mesh_list[j]
        partial_global_qualities = PartialGlobalQualities(
            partial_global_mesh.nx, partial_global_mesh.nz
        )

        # Process each point along this latitude slice
        for k in range(len(partial_global_mesh.x)):
            partial_global_surface_depths = PartialGlobalSurfaceDepths(
                len(global_surfaces)
            )

            partial_basin_surface_depths_list = [
                PartialBasinSurfaceDepths(basin_data) for basin_data in basin_data_list
            ]
            qualities_vector = QualitiesVector(partial_global_mesh.nz)

            basin_indices = in_basin_mesh.get_basin_membership(
                k, j
            )  # List of basin indices
            in_basin_list = [
                InBasin(basin_data, len(global_mesh.z))
                for basin_data in basin_data_list
            ]
            # Mark which basins this point belongs to
            for basin_idx in basin_indices:
                if basin_idx >= 0:
                    in_basin_list[basin_idx].in_basin_lat_lon = True

            if smoothing:
                # Placeholder for future smoothing implementation
                logger.log(
                    "Smoothing option selected but not implemented", logger.DEBUG
                )
                pass
            else:
                # Extract properties for this column and assign velocities/density
                mesh_vector = extract_mesh_vector(partial_global_mesh, k)
                try:
                    qualities_vector.assign_qualities(
                        cvm_registry,
                        velo_mod_1d_data,
                        nz_tomography_data,
                        global_surfaces,
                        basin_data_list,
                        mesh_vector,
                        partial_global_surface_depths,
                        partial_basin_surface_depths_list,
                        in_basin_list,
                        in_basin_mesh,
                        vm_params["topo_type"],
                    )

                    partial_global_qualities.rho[k] = qualities_vector.rho
                    partial_global_qualities.vp[k] = qualities_vector.vp
                    partial_global_qualities.vs[k] = qualities_vector.vs

                    try:
                        temp_inbasin = qualities_vector.inbasin
                    except Exception as e:
                        logger.log(
                            f"Error accessing inbasin attribute: {e}", logger.ERROR
                        )
                        logger.log(
                            f"QualitiesVector object: {qualities_vector}", logger.DEBUG
                        )
                        raise

                    partial_global_qualities.inbasin[k] = temp_inbasin
                except Exception as e:
                    logger.log(
                        f"Error processing point at j={j}, k={k}: {e}", logger.ERROR
                    )
                    raise

        # Write this latitude slice to disk
        write_global_qualities(
            out_dir,
            partial_global_mesh,
            partial_global_qualities,
            j,
            vm_params["min_vs"],
            logger,
        )

    logger.log("Generation of velocity model 100% complete", logger.INFO)
    logger.log(
        f"Model (version: {vm_params['model_version']}) successfully generated and written to {out_dir}",
        logger.INFO,
    )
    elapsed_time = time.time() - start_time
    logger.log(
        f"Velocity model generation completed in {elapsed_time:.2f} seconds",
        logger.INFO,
    )


@cli.from_docstring(app)
def empty_command() -> None:
    """
    Empty command for testing purposes.
    Returns
    -------
    None
    """
    pass


def parse_nzvm_config(config_path: Path) -> dict:
    """
    Parse the NZVM config file and convert it to the dictionary format.

    Parameters
    ----------
    config_path : Path
        Path to the nzvm.cfg file

    Returns
    -------
    dict
        Dictionary containing the model parameters
    """
    vm_params = {}

    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Map NZVM keys to vm_params keys
            if key == "MODEL_VERSION":
                # Always store MODEL_VERSION as string
                vm_params["model_version"] = value
            elif key == "OUTPUT_DIR":
                vm_params["output_dir"] = value
            elif key == "CALL_TYPE":
                vm_params["call_type"] = value
            elif key == "TOPO_TYPE":
                try:
                    vm_params["topo_type"] = TopoTypes[value]
                except KeyError:
                    VMLogger.error(f"Invalid topo type {value}")
                    raise KeyError(f"Invalid topo type {value}")
            else:
                # Convert numeric values
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(
                        f"Numeric value is required for key {key}: {value}"
                    )
                else:
                    # Map NZVM keys to vm_params keys
                    if key == "ORIGIN_LAT":
                        vm_params["origin_lat"] = value
                    elif key == "ORIGIN_LON":
                        vm_params["origin_lon"] = value
                    elif key == "ORIGIN_ROT":
                        vm_params["origin_rot"] = value  # degrees
                    elif key == "EXTENT_X":
                        vm_params["extent_x"] = value  # km
                    elif key == "EXTENT_Y":
                        vm_params["extent_y"] = value  # km
                    elif key == "EXTENT_ZMAX":
                        vm_params["extent_zmax"] = value  # km
                    elif key == "EXTENT_ZMIN":
                        vm_params["extent_zmin"] = value  # km
                    elif key == "EXTENT_Z_SPACING":
                        vm_params["h_depth"] = value  # km
                    elif key == "EXTENT_LATLON_SPACING":
                        vm_params["h_lat_lon"] = value  # km
                    else:
                        # Store any other parameters with lowercase key
                        vm_params[key.lower()] = value

    # Calculate nx, ny, nz based on spacing and extent
    vm_params["nx"] = int(round(vm_params["extent_x"] / vm_params["h_lat_lon"]))
    vm_params["ny"] = int(round(vm_params["extent_y"] / vm_params["h_lat_lon"]))
    vm_params["nz"] = int(
        round(
            (vm_params["extent_zmax"] - vm_params["extent_zmin"]) / vm_params["h_depth"]
        )
    )

    return vm_params
