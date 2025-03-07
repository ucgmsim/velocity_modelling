"""
Velocity Model Generator

This script generates a 3D seismic velocity model based on input parameters.
It creates a mesh grid representation of subsurface properties (P-wave velocity,
S-wave velocity, and density) by combining global velocity models, regional tomographic
models, and local basin models.

Example usage:
    python  generate_velocity_model.py /path/to/nzvm.cfg --out_dir /path/to/output_dir

    # With custom registry location:
    python  generate_velocity_model.py /path/to/nzvm.cfg --out_dir /path/to/output_dir --nzvm_registry /path/to/registry.yaml

    # With specific log level:
    python  generate_velocity_model.py /path/to/nzvm.cfg --out_dir /path/to/output_dir --log-level DEBUG

    # With specific output format:
    python  generate_velocity_model.py /path/to/nzvm.cfg --out_dir /path/to/output_dir --output_format csv  (default: emod3d)


    if --out_dir is not specified, it will use OUTPUT_DIR specified in nzvm.cfg

SAMPLE nzvm.cfg
---------------------
CALL_TYPE=GENERATE_VELOCITY_MOD
MODEL_VERSION=2.07
OUTPUT_DIR=/tmp
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

"""

# import concurrent.futures

import argparse
import sys
import time
from pathlib import Path

from velocity_modelling.cvm.basin_model import (
    InBasin,
    InBasinGlobalMesh,
    PartialBasinSurfaceDepths,
)
from velocity_modelling.cvm.constants import (
    NZVM_REGISTRY_PATH,
)
from velocity_modelling.cvm.geometry import (
    GlobalMesh,
    ModelExtent,
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


def generate_velocity_model(
    cvm_registry: CVMRegistry,
    out_dir: Path,
    vm_params: dict,
    logger: VMLogger,
    smoothing: bool = False,
    progress_interval: int = 5,
    output_format: str = "emod3d",
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
    cvm_registry : CVMRegistry
        Registry containing paths to all required data files.
    out_dir : Path
        Output directory where model files will be written.
    vm_params : dict
        Velocity model parameters (dimensions, resolution, etc).
    logger : VMLogger
        Logger for reporting progress and errors.
    smoothing : bool, optional
        Whether to apply smoothing at model boundaries (default False).
    progress_interval : int, optional
        How often (in %) to log progress updates (default 5%).
    output_format : str, optional
    Format to write the output. Options: "emod3d", "csv"
    """
    # Import the appropriate writer based on format
    logger.log(f"Beginning velocity model generation in {out_dir}", logger.INFO)
    logger.log(f"Model parameters: {vm_params['model_version']}", logger.INFO)
    logger.log(f"Using output format: {output_format}", logger.INFO)

    # Import the appropriate writer based on format
    if output_format == "emod3d":
        from velocity_modelling.cvm.write.emod3d import (
            write_global_qualities,
        )

        logger.log("Using EMOD3D writer module", logger.DEBUG)
    elif output_format == "csv":
        try:
            from velocity_modelling.cvm.write.csv import (
                write_global_qualities,
            )

            logger.log("Using CSV writer module", logger.DEBUG)
        except ImportError:
            logger.log("CSV writer module not found. Creating it now.", logger.WARNING)
            from velocity_modelling.cvm.write.emod3d import (
                write_global_qualities,
            )

            logger.log(
                "Temporarily using EMOD3D writer until CSV writer is implemented",
                logger.WARNING,
            )
    else:
        logger.log(f"Unsupported output format: {output_format}", logger.ERROR)
        raise ValueError(f"Unsupported output format: {output_format}")

    # Create model grid
    logger.log("Generating model grid", logger.INFO)
    model_extent = ModelExtent(vm_params)
    global_mesh = gen_full_model_grid_great_circle(model_extent, logger)
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
                len(global_surfaces.surfaces)
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
            vm_params,
            j,
            logger,
        )

    logger.log("Generation of velocity model 100% complete", logger.INFO)
    logger.log(f"Model successfully generated and written to {out_dir}", logger.INFO)


def parse_nzvm_config(config_path: Path) -> dict:
    """
    Parse the NZVM config file and convert it to the vm_params dictionary format.

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
                vm_params["topo_type"] = value
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
                        vm_params["origin_rot"] = value
                    elif key == "EXTENT_X":
                        vm_params["extent_x"] = value
                    elif key == "EXTENT_Y":
                        vm_params["extent_y"] = value
                    elif key == "EXTENT_ZMAX":
                        vm_params["extent_zmax"] = value
                    elif key == "EXTENT_ZMIN":
                        vm_params["extent_zmin"] = value
                    elif key == "EXTENT_Z_SPACING":
                        vm_params["h_depth"] = value
                    elif key == "EXTENT_LATLON_SPACING":
                        vm_params["h_lat_lon"] = value
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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the velocity model generator.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments containing:
        - config_file: Path to nzvm.cfg configuration file
        - out_dir: Path to output directory (optional, overrides config)
        - nzvm_registry: Path to registry file (optional)
        - log_level: Logging level (optional)
    """
    parser = argparse.ArgumentParser(
        description="Generate a 3D seismic velocity model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_file", type=Path, help="Path to the nzvm.cfg configuration file"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Path to the output directory (overrides config file)",
    )
    parser.add_argument(
        "--nzvm_registry",
        type=Path,
        help="Path to the model registry file",
        default=NZVM_REGISTRY_PATH,
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    parser.add_argument(
        "--output_format",
        choices=["emod3d", "csv"],
        default="emod3d",
        help="Format for the output velocity model",
    )

    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Configure logging
    logger = VMLogger(level=args.log_level)
    logger.log(f"Logger initialized with level {args.log_level}", logger.DEBUG)

    # Validate input files
    config_path = args.config_file
    if not config_path.exists():
        logger.log(f"Configuration file not found: {config_path}", logger.ERROR)
        sys.exit(1)

    if not args.nzvm_registry.exists():
        logger.log(f"Registry file not found: {args.nzvm_registry}", logger.ERROR)
        sys.exit(1)

    # Parse the config file
    try:
        vm_params = parse_nzvm_config(config_path)
        # Validate CALL_TYPE
        if vm_params.get("call_type") != "GENERATE_VELOCITY_MOD":
            logger.log(
                f"Unsupported CALL_TYPE: {vm_params.get('call_type')}", logger.ERROR
            )
            sys.exit(1)
    except Exception as e:
        logger.log(f"Failed to parse config file: {e}", logger.ERROR)
        sys.exit(1)

    # Override output directory if specified
    if args.out_dir:
        out_dir = args.out_dir.resolve()
    else:
        # Use output_dir from config if it exists, otherwise use current directory
        out_dir = Path(vm_params.get("output_dir", "./")).resolve()

    # Prepare output directory
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.log(f"Output will be written to: {out_dir}")

    # Initialize registry and generate model
    try:
        cvm_registry = CVMRegistry(
            vm_params["model_version"],
            logger,
            args.nzvm_registry,
        )
        generate_velocity_model(
            cvm_registry, out_dir, vm_params, logger, output_format=args.output_format
        )

        elapsed_time = time.time() - start_time
        logger.log(
            f"Velocity model generation completed in {elapsed_time:.2f} seconds",
            logger.INFO,
        )
    except Exception as e:
        logger.log(f"Model generation failed: {e}", logger.ERROR)
        raise
