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
    # Basic usage: to use everything defined in nzcvm.cfg
    nzcvm generate-velocity-model /path/to/nzcvm.cfg

    # To override output directory
    nzcvm generate-velocity-model /path/to/nzcvm.cfg --out-dir /path/to/output_dir

    # To override DATA_ROOT directory
    nzcvm generate-velocity-model /path/to/nzcvm.cfg --data-root /custom/data/path

    # To override "MODEL_VERSION" in nzcvm.cfg to use a .yaml file for a custom model version.
    # Requires a .yaml file with the model version under the "model_versions" directory. (eg. 2p07.yaml for model version 2.07)
    nzcvm generate-velocity-model /path/to/nzcvm.cfg --model-version 2.07

    # With custom registry location:
    nzcvm generate-velocity-model /path/to/nzcvm.cfg --nzcvm-registry /path/to/registry.yaml

    # With specific log level:
    nzcvm generate-velocity-model /path/to/nzcvm.cfg --log-level DEBUG

    # With specific output format:
    nzcvm generate-velocity-model /path/to/nzcvm.cfg --output-format CSV  (default: EMOD3D)
    nzcvm generate-velocity-model /path/to/nzcvm.cfg --output-format HDF5

    [example] nzcvm.cfg
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

import logging
import sys
import time
from logging import Logger
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from qcore import cli
from velocity_modelling.basin_model import (
    InBasin,
    InBasinGlobalMesh,
    PartialBasinSurfaceDepths,
)
from velocity_modelling.constants import (
    DATA_ROOT,
    NZCVM_REGISTRY_PATH,
    TopoTypes,
    WriteFormat,
)
from velocity_modelling.geometry import (
    GlobalMesh,
    MeshVector,
    gen_full_model_grid_great_circle,
)
from velocity_modelling.global_model import PartialGlobalSurfaceDepths
from velocity_modelling.registry import CVMRegistry
from velocity_modelling.velocity3d import PartialGlobalQualities, QualitiesVector

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nzcvm")

app = typer.Typer(pretty_exceptions_enable=False)


def write_velo_mod_corners_text_file(
    global_mesh: GlobalMesh, output_dir: Path | str, logger: logging.Logger
) -> None:
    """
    Write velocity model corners to a text file for reference and visualization.

    Records the geographic coordinates of model corners (min/max latitude and longitude)
    for later reference and plotting.

    Parameters
    ----------
    global_mesh : GlobalMesh
        Object containing the global mesh data with longitude and latitude arrays.
    output_dir : Path or str
        Directory where the log file will be saved.
    logger : Logger
        Logger for reporting progress and errors.

    Raises
    ------
    OSError
        If the file cannot be written due to permissions or disk issues.
    """
    log_file_name = Path(output_dir) / "Log" / "VeloModCorners.txt"
    log_file_name.parent.mkdir(parents=True, exist_ok=True)

    nx = len(global_mesh.x)
    ny = len(global_mesh.y)

    logger.log(logging.INFO, f"Writing velocity model corners to {log_file_name}")
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
        logger.log(logging.INFO, "Velocity model corners file write complete.")
    except OSError as e:
        logger.log(logging.ERROR, f"Failed to write velocity model corners: {e}")
        raise OSError(
            f"Failed to write velocity model corners to {log_file_name}: {str(e)}"
        )


@cli.from_docstring(app)
def generate_3d_model(
    nzcvm_cfg_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out_dir: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    nzcvm_registry: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = NZCVM_REGISTRY_PATH,
    model_version: Annotated[str | None, typer.Option()] = None,
    output_format: Annotated[str, typer.Option()] = WriteFormat.EMOD3D.name,
    data_root: Annotated[
        Path,
        typer.Option(
            file_okay=False,
            exists=True,
            help="Override the default DATA_ROOT directory",
        ),
    ] = DATA_ROOT,
    smoothing: Annotated[
        bool, typer.Option()
    ] = False,  # placeholder for smoothing, not implemented yet
    log_level: Annotated[str, typer.Option()] = "INFO",
) -> None:
    """
    Generate a 3D velocity model and write it to disk.

    This is the main function that orchestrates the velocity model generation process:
    1. Creates the model mesh grid
    2. Loads all required datasets (global models, tomography, basins)
    3. Processes each latitude slice and populates velocity/density values
    4. Writes results to disk in the specified format

    Parameters
    ----------
    nzcvm_cfg_path : Path
        Path to the nzcvm.cfg configuration file.
    out_dir : Path, optional
        Path to the output directory where the velocity model files will be written (overrides OUTPUT_DIR in config file).
        If not provided, the directory specified in the config file will be used.
    nzcvm_registry : Path, optional
        Path to the model registry file (default: NZCVM_REGISTRY_PATH).
    model_version : str, optional
        Version of the model to use (overrides MODEL_VERSION in config file).
        If not provided, the version from the config file will be used.
    output_format : str, optional
        Format to write the output. Options: "EMOD3D", "CSV", "HDF5" (default: "EMOD3D").
    data_root : Path, optional
        Override the default DATA_ROOT directory (default: derived from constants.py).
    smoothing : bool, optional
        Unsupported option for future smoothing implementation at model boundaries (default: False).

    log_level : str, optional
        Logging level for the script (default: "INFO").

    Raises
    ------
    ValueError
        If the config file is invalid, the output format is unsupported, or CALL_TYPE is incorrect.
    OSError
        If there are issues writing to the output directory or files.
    RuntimeError
        If an error occurs during model generation or data processing.
    """
    start_time = time.time()

    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    logger.log(logging.DEBUG, f"Logger initialized with level {log_level}")
    logger.log(logging.INFO, "Beginning velocity model generation")

    # Override DATA_ROOT if provided

    data_root = data_root.resolve()
    logger.log(logging.INFO, f"data_root set to {data_root}")

    # Validate DATA_ROOT (default or overridden)
    if not data_root.exists():
        logger.log(logging.ERROR, f"data_root path does not exist: {data_root}")
        raise ValueError(f"data_root path does not exist: {data_root}")
    if not data_root.is_dir():
        logger.log(logging.ERROR, f"data_root is not a directory: {data_root}")
        raise ValueError(f"data_root is not a directory: {data_root}")
    logger.log(logging.DEBUG, f"data_root validated as {data_root}")

    # Parse the config file
    try:
        vm_params = parse_nzcvm_config(nzcvm_cfg_path, logger)
        if vm_params.get("call_type") != "GENERATE_VELOCITY_MOD":
            logger.log(
                logging.ERROR, f"Unsupported CALL_TYPE: {vm_params.get('call_type')}"
            )
            raise ValueError(f"Unsupported CALL_TYPE: {vm_params.get('call_type')}")
    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            raise  # Re-raise critical exceptions
        logger.log(logging.ERROR, f"Failed to parse config file: {e}")
        raise ValueError(f"Failed to parse config file {nzcvm_cfg_path}: {str(e)}")

    # If out_dir is not provided, use output_dir from vm_params if available
    if out_dir is None:
        if "output_dir" in vm_params:
            out_dir = Path(vm_params["output_dir"])
            logger.log(logging.INFO, f"Using output_dir from config: {out_dir}")
        else:
            logger.log(logging.ERROR, "No output directory specified")
            raise ValueError("No output directory specified in config or command line")

    logger.log(logging.INFO, f"Output directory set to {out_dir}")

    # Use --model-version if provided, otherwise fall back to MODEL_VERSION from config
    if model_version and model_version != vm_params.get("model_version"):
        logger.log(
            logging.INFO,
            f"Updating model_version from {vm_params['model_version']} to {model_version}",
        )
        vm_params["model_version"] = model_version
    else:
        logger.log(logging.INFO, f"Using model version: {vm_params['model_version']}")

    # Validate and import the appropriate writer based on format
    try:
        _ = WriteFormat[output_format.upper()]
    except KeyError:
        logger.log(logging.ERROR, f"Unsupported output format: {output_format}")
        raise ValueError(f"Unsupported output format: {output_format}")

    import importlib

    try:
        module_name = f"velocity_modelling.write.{output_format.lower()}"
        writer_module = importlib.import_module(module_name)
        write_global_qualities = writer_module.write_global_qualities
        logger.log(logging.DEBUG, f"Using {output_format} writer module")
    except ImportError as e:
        logger.log(
            logging.ERROR, f"Failed to import writer module for {output_format}: {e}"
        )
        raise ValueError(f"Unsupported output format: {output_format}")

    # Ensure output directory exists
    out_dir = out_dir.resolve()
    try:
        out_dir.mkdir(exist_ok=True, parents=True)
    except OSError as e:
        logger.log(logging.ERROR, f"Failed to create output directory {out_dir}: {e}")
        raise OSError(f"Failed to create output directory {out_dir}: {str(e)}")

    # Initialize registry and generate model
    cvm_registry = CVMRegistry(
        vm_params["model_version"], data_root, nzcvm_registry, logger
    )

    # Create model grid
    logger.log(logging.INFO, "Generating model grid")
    global_mesh = gen_full_model_grid_great_circle(vm_params, logger)
    write_velo_mod_corners_text_file(global_mesh, out_dir, logger)

    # Load all required data
    logger.log(logging.INFO, "Loading model data")
    try:
        vm1d_data, nz_tomography_data, global_surfaces, basin_data_list = (
            cvm_registry.load_all_global_data()
        )
    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            raise  # Re-raise critical exceptions
        logger.log(logging.ERROR, f"Failed to load model data: {e}")
        raise RuntimeError(f"Failed to load model data: {str(e)}")

    # Preprocess basin membership for efficiency
    logger.log(logging.INFO, "Pre-processing basin membership")
    try:
        in_basin_mesh, partial_global_mesh_list = (
            InBasinGlobalMesh.preprocess_basin_membership(
                global_mesh,
                basin_data_list,
                logger,
                smooth_bound=nz_tomography_data.smooth_boundary,
            )
        )
    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            raise  # Re-raise critical exceptions
        logger.log(logging.ERROR, f"Error preprocessing basin membership: {e}")
        raise RuntimeError(f"Error preprocessing basin membership: {str(e)}")

    # Process each latitude slice
    total_slices = len(global_mesh.y)
    for j in tqdm(range(total_slices), desc="Generating velocity model", unit="slice"):
        partial_global_mesh = partial_global_mesh_list[j]
        partial_global_qualities = PartialGlobalQualities(
            partial_global_mesh.nx, partial_global_mesh.nz
        )

        for k in range(len(partial_global_mesh.x)):
            partial_global_surface_depths = PartialGlobalSurfaceDepths(
                len(global_surfaces)
            )
            partial_basin_surface_depths_list = [
                PartialBasinSurfaceDepths(basin_data) for basin_data in basin_data_list
            ]
            qualities_vector = QualitiesVector(partial_global_mesh.nz)

            basin_indices = in_basin_mesh.get_basin_membership(k, j)
            in_basin_list = [
                InBasin(basin_data, len(global_mesh.z))
                for basin_data in basin_data_list
            ]
            for basin_idx in basin_indices:
                if basin_idx >= 0:
                    in_basin_list[basin_idx].in_basin_lat_lon = True

            if smoothing:
                logger.log(
                    logging.DEBUG, "Smoothing option selected but not implemented"
                )
                # Placeholder for future implementation
            else:
                try:
                    mesh_vector = MeshVector(partial_global_mesh, k)
                    qualities_vector.assign_qualities(
                        cvm_registry,
                        vm1d_data,
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
                    partial_global_qualities.inbasin[k] = qualities_vector.inbasin
                except AttributeError as e:
                    logger.log(
                        logging.ERROR,
                        f"Error accessing qualities vector attributes at j={j}, k={k}: {e}",
                    )
                    raise RuntimeError(
                        f"Error processing point at j={j}, k={k}: {str(e)}"
                    )
                except Exception as e:
                    if isinstance(e, (SystemExit, KeyboardInterrupt)):
                        raise  # Re-raise critical exceptions
                    logger.log(
                        logging.ERROR, f"Error processing point at j={j}, k={k}: {e}"
                    )
                    raise RuntimeError(
                        f"Error processing point at j={j}, k={k}: {str(e)}"
                    )

        # Write this latitude slice to disk
        try:
            # Use a single function call format for all writer modules
            write_global_qualities(
                out_dir,
                partial_global_mesh,
                partial_global_qualities,
                j,
                vm_params,
                logger,
            )
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise  # Re-raise critical exceptions
            logger.log(logging.ERROR, f"Error writing slice j={j}: {e}")
            raise OSError(f"Failed to write slice j={j} to {out_dir}: {str(e)}")

    logger.log(logging.INFO, "Generation of velocity model 100% complete")
    logger.log(
        logging.INFO,
        f"Model (version: {vm_params['model_version']}) successfully generated and written to {out_dir}",
    )
    elapsed_time = time.time() - start_time
    logger.log(
        logging.INFO,
        f"Velocity model generation completed in {elapsed_time:.2f} seconds",
    )


def parse_nzcvm_config(config_path: Path, logger: Logger | None = None) -> dict:
    """
    Parse the nzcvm config file and convert it to a dictionary format.

    Parameters
    ----------
    config_path : Path
        Path to the nzcvm.cfg file.
    logger : Logger, optional
        Logger instance for logging messages.

    Returns
    -------
    dict
        Dictionary containing the model parameters.

    Raises
    ------
    FileNotFoundError
        If the config file cannot be found.
    ValueError
        If a numeric value is expected but not provided, or if parsing fails.
    KeyError
        If an invalid TOPO_TYPE is specified.
    """
    if logger is None:
        logger = Logger(name="velocity_model.parse_nzcvm_config")

    vm_params = {}
    numeric_keys = {
        "ORIGIN_LAT",
        "ORIGIN_LON",
        "ORIGIN_ROT",
        "EXTENT_X",
        "EXTENT_Y",
        "EXTENT_ZMAX",
        "EXTENT_ZMIN",
        "EXTENT_Z_SPACING",
        "EXTENT_LATLON_SPACING",
        "MIN_VS",
    }
    string_keys = {"MODEL_VERSION", "OUTPUT_DIR", "CALL_TYPE"}
    key_mapping = {"EXTENT_Z_SPACING": "h_depth", "EXTENT_LATLON_SPACING": "h_lat_lon"}

    try:
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                key, value = map(str.strip, line.split("=", 1))
                float_value = None
                try:
                    float_value = float(value)
                except ValueError:
                    pass
                dest_key = key_mapping.get(key, key.lower())

                if key in string_keys:
                    vm_params[dest_key] = value
                elif key == "TOPO_TYPE":
                    try:
                        vm_params[dest_key] = TopoTypes[value]
                    except KeyError:
                        Logger.error(f"Invalid topo type {value}")
                        raise KeyError(f"Invalid topo type {value}")
                elif key in numeric_keys:
                    if float_value is None:
                        raise ValueError(
                            f"Numeric value required for key {key}: {value}"
                        )
                    vm_params[dest_key] = float_value
                else:
                    vm_params[dest_key] = (
                        float_value if float_value is not None else value
                    )

        # Calculate nx, ny, nz based on spacing and extent
        # Python round() behaves differently than C/C++ round(), rounds to the nearest even number
        # eg. round(1.5)->2 round(2.5)->2, round(3.5)->4
        # Adding 0.5 and casting with int() ensures the same behaviour as C round()

        vm_params["nx"] = int(vm_params["extent_x"] / vm_params["h_lat_lon"] + 0.5)
        vm_params["ny"] = int(vm_params["extent_y"] / vm_params["h_lat_lon"] + 0.5)
        vm_params["nz"] = int(
            (vm_params["extent_zmax"] - vm_params["extent_zmin"]) / vm_params["h_depth"]
            + 0.5
        )
    except FileNotFoundError:
        logger.log(logging.ERROR, "Config file {config_path} not found")
        raise FileNotFoundError(f"Config file {config_path} not found")
    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            raise  # Re-raise critical exceptions
        logger.log(logging.ERROR, f"Error parsing config file {config_path}: {e}")
        raise ValueError(f"Error parsing config file {config_path}: {str(e)}")

    return vm_params


if __name__ == "__main__":
    app()
