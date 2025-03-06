"""
Velocity Model Generator

This script generates a 3D seismic velocity model based on input parameters.
It creates a mesh grid representation of subsurface properties (P-wave velocity,
S-wave velocity, and density) by combining global velocity models, regional tomographic
models, and local basin models.

Example usage:
    python  generate_velocity_model.py /path/to/vm_params.yaml /path/to/output_dir

    # With custom registry location:
     python  generate_velocity_model.py /path/to/vm_params.yaml /path/to/output_dir --nzvm_registry /path/to/registry.yaml

    # With specific log level:
     python  generate_velocity_model.py /path/to/vm_params.yaml /path/to/output_dir--log-level DEBUG
"""

# import concurrent.futures

import time
from pathlib import Path
import argparse
import yaml
import sys
from typing import Dict

from velocity_modelling.cvm.constants import NZVM_REGISTRY_PATH
from velocity_modelling.cvm.basin_model import (
    InBasin,
    PartialBasinSurfaceDepths,
    InBasinGlobalMesh,
)
from velocity_modelling.cvm.geometry import (
    gen_full_model_grid_great_circle,
    extract_mesh_vector,
    GlobalMesh,
    ModelExtent,
)
from velocity_modelling.cvm.global_model import PartialGlobalSurfaceDepths
from velocity_modelling.cvm.logging import VMLogger
from velocity_modelling.cvm.registry import CVMRegistry
from velocity_modelling.cvm.velocity3d import PartialGlobalQualities, QualitiesVector
from velocity_modelling.cvm.write.emod3d import write_global_qualities


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
    vm_params: Dict,
    logger: VMLogger,
    smoothing: bool = False,
    progress_interval: int = 5,
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
    vm_params : Dict
        Velocity model parameters (dimensions, resolution, etc).
    logger : VMLogger
        Logger for reporting progress and errors.
    smoothing : bool, optional
        Whether to apply smoothing at model boundaries (default False).
    progress_interval : int, optional
        How often (in %) to log progress updates (default 5%).
    """
    logger.log(f"Beginning velocity model generation in {out_dir}", logger.INFO)
    logger.log(f"Model parameters: {vm_params['model_version']}", logger.INFO)

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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the velocity model generator.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments containing:
        - vm_params: Path to velocity model parameters file
        - out_dir: Path to output directory
        - nzvm_registry: Path to registry file (optional)
        - log_level: Logging level (optional)
    """
    parser = argparse.ArgumentParser(
        description="Generate a 3D seismic velocity model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("vm_params", type=Path, help="Path to the vm_params.yaml file")
    parser.add_argument("out_dir", type=Path, help="Path to the output directory")
    parser.add_argument(
        "--nzvm_registry",
        type=Path,
        help="Path to the model registry file",
        default=NZVM_REGISTRY_PATH,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
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
    vm_params_path = args.vm_params
    if not vm_params_path.exists():
        logger.log(f"VM params file not found: {vm_params_path}", logger.ERROR)
        sys.exit(1)

    if not args.nzvm_registry.exists():
        logger.log(f"Registry file not found: {args.nzvm_registry}", logger.ERROR)
        sys.exit(1)

    # Prepare output directory
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(exist_ok=True, parents=True)
    logger.log(f"Output will be written to: {out_dir}")

    # Load velocity model parameters
    logger.log(f"Using vm_params file: {vm_params_path}")
    try:
        with open(vm_params_path, "r") as f:
            vm_params = yaml.safe_load(f)
    except Exception as e:
        logger.log(f"Failed to load vm_params file: {e}", logger.ERROR)
        sys.exit(1)

    # Initialize registry and generate model
    try:
        cvm_registry = CVMRegistry(
            vm_params["model_version"],
            logger,
            args.nzvm_registry,
        )
        generate_velocity_model(cvm_registry, out_dir, vm_params, logger)

        elapsed_time = time.time() - start_time
        logger.log(
            f"Velocity model generation completed in {elapsed_time:.2f} seconds",
            logger.INFO,
        )
    except Exception as e:
        logger.log(f"Model generation failed: {e}", logger.ERROR)
        sys.exit(1)
