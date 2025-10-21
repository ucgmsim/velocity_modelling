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
    nzcvm generate-3d-model /path/to/nzcvm.cfg

    # To override output directory
    nzcvm generate-3d-model /path/to/nzcvm.cfg --out-dir /path/to/output_dir

    # To override DATA_ROOT directory
    nzcvm generate-3d-model /path/to/nzcvm.cfg --data-root /custom/data/path

    # To override "MODEL_VERSION" in nzcvm.cfg to use a .yaml file for a custom model version.
    # Requires a .yaml file with the model version under the "model_versions" directory. (eg. 2p07.yaml for model version 2.07)
    nzcvm generate-3d-model /path/to/nzcvm.cfg --model-version 2.07

    # With custom registry location:
    nzcvm generate-3d-model /path/to/nzcvm.cfg --nzcvm-registry /path/to/registry.yaml

    # With specific log level:
    nzcvm generate-3d-model /path/to/nzcvm.cfg --log-level DEBUG

    # With specific output format:
    nzcvm generate-3d-model /path/to/nzcvm.cfg --output-format CSV  (default: EMOD3D)
    nzcvm generate-3d-model /path/to/nzcvm.cfg --output-format HDF5

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
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import Logger
from multiprocessing import Queue
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from tqdm import tqdm

from qcore import cli
from velocity_modelling.basin_model import (
    BasinData,
    BasinMembership,
    InBasin,
    PartialBasinSurfaceDepths,
)
from velocity_modelling.constants import (
    TopoTypes,
    WriteFormat,
    get_data_root,
)
from velocity_modelling.geometry import (
    GlobalMesh,
    MeshVector,
    PartialGlobalMesh,
    gen_full_model_grid_great_circle,
)
from velocity_modelling.global_model import (
    GlobalSurfaceRead,
    PartialGlobalSurfaceDepths,
    TomographyData,
)
from velocity_modelling.registry import CVMRegistry
from velocity_modelling.velocity1d import VelocityModel1D
from velocity_modelling.velocity3d import PartialGlobalQualities, QualitiesVector

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nzcvm")

app = typer.Typer(pretty_exceptions_enable=False)


# ============================================================================
# Configuration Loading Functions
# ============================================================================
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
        If a numeric value is expected but not provided, if parsing fails.
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
                        logger.log(logging.ERROR, f"Invalid topo type {value}")
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
        logger.log(logging.ERROR, f"Config file {config_path} not found")
        raise FileNotFoundError(f"Config file {config_path} not found")
    except (ValueError, KeyError) as e:
        logger.log(logging.ERROR, f"Error parsing config file {config_path}: {e}")
        raise
    except KeyboardInterrupt:
        raise
    except (OSError, PermissionError) as e:
        logger.log(logging.ERROR, f"Error parsing config file {config_path}: {e}")
        raise ValueError(f"Error parsing config file {config_path}: {str(e)}")

    return vm_params


def _load_vm_params_from_cfg(
    nzcvm_cfg_path: Path,
    out_dir: Path | None = None,
    nzcvm_data_root: Path | None = None,
    model_version: str | None = None,
) -> dict:
    """
    Load velocity model parameters from configuration file.

    Parameters
    ----------
    nzcvm_cfg_path : Path
        Path to the NZCVM configuration file
    out_dir : Path, optional
        Override output directory
    nzcvm_data_root : Path, optional
        Override data root directory
    model_version : str, optional
        Override model version

    Returns
    -------
    dict
        Dictionary of velocity model parameters
    """
    # Parse the config file using the existing parser
    vm_params = parse_nzcvm_config(nzcvm_cfg_path, logger)

    # Check call type
    if vm_params.get("call_type") != "GENERATE_VELOCITY_MOD":
        raise ValueError(f"Unsupported CALL_TYPE: {vm_params.get('call_type')}")

    # Apply overrides if provided
    if out_dir is not None:
        vm_params["output_dir"] = out_dir
    if model_version is not None:
        vm_params["model_version"] = model_version

    # Ensure output_dir is a Path object
    vm_params["output_dir"] = Path(vm_params["output_dir"])

    return vm_params


# ============================================================================
# Common Processing Functions (shared between serial and parallel)
# ============================================================================


def _compute_point_qualities(
    k: int,
    j: int,
    partial_mesh: PartialGlobalMesh,
    global_mesh: GlobalMesh,
    basin_membership: BasinMembership,
    cvm_registry: CVMRegistry,
    vm1d_data: VelocityModel1D,
    nz_tomography_data: TomographyData,
    global_surfaces: list[GlobalSurfaceRead],
    basin_data_list: list[BasinData],
    vm_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute velocity model properties for a single point (x,y) at all depths.

    This function contains the core computation logic shared between serial
    and parallel processing modes.

    Parameters
    ----------
    k : int
        X-coordinate index in the partial mesh
    j : int
        Y-coordinate index in the global mesh
    partial_mesh : PartialGlobalMesh
        Mesh data for the current latitude slice
    global_mesh : GlobalMesh
        The complete model mesh grid
    basin_membership : BasinMembership
        Precomputed basin membership data
    cvm_registry : CVMRegistry
        Registry containing model data and configurations
    vm1d_data : VelocityModel1D
        1D velocity model data
    nz_tomography_data : TomographyData
        Tomography model data for New Zealand
    global_surfaces : list[GlobalSurfaceRead]
        List of global surface models
    basin_data_list : list[BasinData]
        List of basin model data
    vm_params : dict
        Velocity model parameters

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Arrays of (rho, vp, vs, inbasin) for all depths at this point
    """
    # Initialize depth-dependent data structures
    partial_global_surface_depths = PartialGlobalSurfaceDepths(len(global_surfaces))
    partial_basin_surface_depths_list = [
        PartialBasinSurfaceDepths(basin_data) for basin_data in basin_data_list
    ]
    qualities_vector = QualitiesVector(partial_mesh.nz)

    # Determine basin membership for this point
    basin_indices = basin_membership.get_basin_membership(k, j)
    in_basin_list = [
        InBasin(basin_data, len(global_mesh.z)) for basin_data in basin_data_list
    ]
    for basin_idx in basin_indices:
        if basin_idx >= 0:
            in_basin_list[basin_idx].in_basin_lat_lon = True

    # Compute velocity model properties at this point
    mesh_vector = MeshVector(partial_mesh, k)
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
        basin_membership,
        vm_params["topo_type"],
    )

    return (
        qualities_vector.rho,
        qualities_vector.vp,
        qualities_vector.vs,
        qualities_vector.inbasin,
    )


def _process_single_slice(
    j: int,
    partial_mesh: PartialGlobalMesh,
    global_mesh: GlobalMesh,
    basin_membership: BasinMembership,
    cvm_registry: CVMRegistry,
    vm1d_data: VelocityModel1D,
    nz_tomography_data: TomographyData,
    global_surfaces: list[GlobalSurfaceRead],
    basin_data_list: list[BasinData],
    vm_params: dict,
    smoothing: bool = False,
    logger: logging.Logger | None = None,
) -> PartialGlobalQualities:
    """
    Process a single y-slice to compute velocity model properties.

    This function processes one latitude slice of the velocity model, computing
    P-wave velocity, S-wave velocity, density, and basin membership for all
    points in the slice. Used by both serial and parallel processing modes.

    Parameters
    ----------
    j : int
        Y-slice index to process
    partial_mesh : PartialGlobalMesh
        Mesh data for this slice
    global_mesh : GlobalMesh
        The complete model mesh grid
    basin_membership : BasinMembership
        Precomputed basin membership data
    cvm_registry : CVMRegistry
        Registry containing model data and configurations
    vm1d_data : VelocityModel1D
        1D velocity model data
    nz_tomography_data : TomographyData
        Tomography model data for New Zealand
    global_surfaces : list[GlobalSurfaceRead]
        List of global surface models
    basin_data_list : list[BasinData]
        List of basin model data
    vm_params : dict
        Velocity model parameters
    smoothing : bool, optional
        Whether to apply smoothing (not yet implemented)
    logger : logging.Logger, optional
        Logger for error reporting

    Returns
    -------
    PartialGlobalQualities
        Computed velocity model properties for this slice
    """
    if logger is None:
        logger = logging.getLogger("nzcvm")

    partial_qualities = PartialGlobalQualities(partial_mesh.nx, partial_mesh.nz)

    # Process each x-coordinate in this y-slice
    for k in range(len(partial_mesh.x)):
        # Apply smoothing if requested (placeholder for future implementation)
        if smoothing:
            logger.log(logging.DEBUG, "Smoothing option selected but not implemented")

        try:
            # Compute velocity model properties for this point
            rho, vp, vs, inbasin = _compute_point_qualities(
                k,
                j,
                partial_mesh,
                global_mesh,
                basin_membership,
                cvm_registry,
                vm1d_data,
                nz_tomography_data,
                global_surfaces,
                basin_data_list,
                vm_params,
            )

            # Store computed properties
            partial_qualities.rho[k] = rho
            partial_qualities.vp[k] = vp
            partial_qualities.vs[k] = vs
            partial_qualities.inbasin[k] = inbasin

        except AttributeError as e:
            logger.log(
                logging.ERROR,
                f"Error accessing qualities vector attributes at j={j}, k={k}: {e}",
            )
            raise RuntimeError(f"Error processing point at j={j}, k={k}: {str(e)}")
        except (ValueError, RuntimeError) as e:
            logger.log(logging.ERROR, f"Error processing point at j={j}, k={k}: {e}")
            raise RuntimeError(f"Error processing point at j={j}, k={k}: {str(e)}")
        except KeyboardInterrupt:
            raise
        except (AttributeError, ValueError, RuntimeError) as e:
            logger.log(
                logging.ERROR, f"Unexpected error processing point at j={j}, k={k}: {e}"
            )
            raise RuntimeError(
                f"Unexpected error processing point at j={j}, k={k}: {str(e)}"
            )

    return partial_qualities


# ============================================================================
# Serial Processing Implementation
# ============================================================================


def _run_serial_processing(
    global_mesh: GlobalMesh,
    basin_membership: BasinMembership,
    partial_global_mesh_list: list[PartialGlobalMesh],
    cvm_registry: CVMRegistry,
    vm1d_data: VelocityModel1D,
    nz_tomography_data: TomographyData,
    global_surfaces: list[GlobalSurfaceRead],
    basin_data_list: list[BasinData],
    vm_params: dict,
    out_dir: Path,
    write_global_qualities: Callable,
    smoothing: bool,
    logger: logging.Logger,
) -> None:
    """
    Execute the complete serial processing workflow.

    This function handles the complete serial processing workflow, iterating
    through each latitude slice and computing velocity model properties in
    a single-threaded manner.

    Parameters
    ----------
    global_mesh : GlobalMesh
        The complete model mesh grid
    basin_membership : BasinMembership
        Precomputed basin membership data
    partial_global_mesh_list : list[PartialGlobalMesh]
        List of partial meshes for each y-slice
    cvm_registry : CVMRegistry
        Registry containing model data and configurations
    vm1d_data : VelocityModel1D
        1D velocity model data
    nz_tomography_data : TomographyData
        Tomography model data for New Zealand
    global_surfaces : list[GlobalSurfaceRead]
        List of global surface models
    basin_data_list : list[BasinData]
        List of basin model data
    vm_params : dict
        Velocity model parameters
    out_dir : Path
        Output directory for results
    write_global_qualities : Callable
        Function to write slice data to disk
    smoothing : bool
        Whether to apply smoothing
    logger : logging.Logger
        Logger for progress reporting
    """
    total_slices = len(global_mesh.y)
    logger.log(logging.INFO, f"Starting serial processing of {total_slices} slices")

    for j in tqdm(range(total_slices), desc="Generating velocity model", unit="slice"):
        partial_mesh = partial_global_mesh_list[j]

        # Process this slice
        partial_qualities = _process_single_slice(
            j,
            partial_mesh,
            global_mesh,
            basin_membership,
            cvm_registry,
            vm1d_data,
            nz_tomography_data,
            global_surfaces,
            basin_data_list,
            vm_params,
            smoothing,
            logger,
        )

        # Write this latitude slice to disk
        try:
            write_global_qualities(
                out_dir, partial_mesh, partial_qualities, j, vm_params, logger
            )
        except KeyboardInterrupt:
            raise
        except (OSError, IOError) as e:
            logger.log(logging.ERROR, f"File system error writing slice {j}: {e}")
            raise OSError(f"Failed to write slice {j}: {str(e)}")
        except (ValueError, RuntimeError) as e:
            logger.log(logging.ERROR, f"Error writing slice {j} to disk: {e}")
            raise RuntimeError(f"Error writing slice {j}: {str(e)}")

    logger.log(logging.INFO, "Serial processing completed successfully")


# ============================================================================
# Parallel Processing Implementation
# ============================================================================

# Worker-visible globals, set once in the parent process for fork context
_CVM_REGISTRY = None
_MODEL_DATA = None  # (vm1d_data, nz_tomo, global_surfaces, basin_data_list)
_MESH_DATA = None  # (global_mesh, basin_membership, partial_global_mesh_list)
_VM_PARAMS = None
_OUTPUT_QUEUE = None


def _init_worker_process(queue: Queue, blas_threads: int) -> None:
    """
    Initialize worker process with output queue and BLAS thread limits.

    This function runs once in each worker process to set up the output queue
    and configure BLAS threading to prevent oversubscription.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue for sending results to the writer process
    blas_threads : int
        Maximum number of BLAS threads per worker process
    """
    global _OUTPUT_QUEUE
    _OUTPUT_QUEUE = queue

    # Configure BLAS threading to prevent oversubscription
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=blas_threads)
    except ImportError:
        # Fallback using environment variables if threadpoolctl unavailable
        os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
        os.environ["OMP_NUM_THREADS"] = str(blas_threads)
        os.environ["OMP_DYNAMIC"] = "FALSE"


def _compute_slice(slice_index: int) -> int:
    """
    Compute a single y-slice and send results to the writer process.

    Parameters
    ----------
    slice_index : int
        Y-slice index to process

    Returns
    -------
    int
        The slice index that was processed (for exception handling)

    Raises
    ------
    RuntimeError
        If worker globals are not properly initialized
    """
    # Verify worker globals are initialized
    if any(x is None for x in (_CVM_REGISTRY, _MODEL_DATA, _MESH_DATA, _VM_PARAMS)):
        raise RuntimeError(
            "Worker globals not initialized; ensure 'fork' context is used."
        )

    # Unpack global data once for efficiency
    cvm_registry = _CVM_REGISTRY
    vm1d_data, nz_tomography_data, global_surfaces, basin_data_list = _MODEL_DATA
    global_mesh, basin_membership, partial_global_mesh_list = _MESH_DATA

    # Process this slice using the common function
    partial_mesh = partial_global_mesh_list[slice_index]
    partial_qualities = _process_single_slice(
        slice_index,
        partial_mesh,
        global_mesh,
        basin_membership,
        cvm_registry,
        vm1d_data,
        nz_tomography_data,
        global_surfaces,
        basin_data_list,
        _VM_PARAMS,
    )

    # Send results to writer process
    _OUTPUT_QUEUE.put((slice_index, partial_mesh, partial_qualities))

    return slice_index


def _setup_parallel_globals(
    cvm_registry: CVMRegistry,
    vm1d_data: VelocityModel1D,
    nz_tomography_data: TomographyData,
    global_surfaces: list[GlobalSurfaceRead],
    basin_data_list: list[BasinData],
    global_mesh: GlobalMesh,
    basin_membership: BasinMembership,
    partial_global_mesh_list: list[PartialGlobalMesh],
    vm_params: dict,
) -> None:
    """
    Set up global variables for parallel worker processes.

    This function initializes the global variables that will be inherited
    by worker processes when using the 'fork' multiprocessing context.

    Parameters
    ----------
    cvm_registry : CVMRegistry
        Registry containing model data and configurations
    vm1d_data : VelocityModel1D
        1D velocity model data
    nz_tomography_data : TomographyData
        Tomography model data for New Zealand
    global_surfaces : list[GlobalSurfaceRead]
        List of global surface models
    basin_data_list : list[BasinData]
        List of basin model data
    global_mesh : GlobalMesh
        The complete model mesh grid
    basin_membership : BasinMembership
        Precomputed basin membership data
    partial_global_mesh_list : list[PartialGlobalMesh]
        List of partial meshes for each y-slice
    vm_params : dict
        Velocity model parameters
    """
    global _CVM_REGISTRY, _MODEL_DATA, _MESH_DATA, _VM_PARAMS

    _CVM_REGISTRY = cvm_registry
    _MODEL_DATA = (vm1d_data, nz_tomography_data, global_surfaces, basin_data_list)
    _MESH_DATA = (global_mesh, basin_membership, partial_global_mesh_list)
    _VM_PARAMS = vm_params


def _run_parallel_processing(
    global_mesh: GlobalMesh,
    basin_membership: BasinMembership,
    partial_global_mesh_list: list[PartialGlobalMesh],
    cvm_registry: CVMRegistry,
    vm1d_data: VelocityModel1D,
    nz_tomography_data: TomographyData,
    global_surfaces: list[GlobalSurfaceRead],
    basin_data_list: list[BasinData],
    vm_params: dict,
    out_dir: Path,
    np_workers: int,
    blas_threads: int,
    logger: logging.Logger,
) -> None:
    """
    Execute the complete parallel processing workflow.

    This function coordinates parallel processing using multiple worker processes
    and a dedicated HDF5 writer process. Each worker processes individual slices.

    Parameters
    ----------
    global_mesh : GlobalMesh
        The complete model mesh grid
    basin_membership : BasinMembership
        Precomputed basin membership data
    partial_global_mesh_list : list[PartialGlobalMesh]
        List of partial meshes for each y-slice
    cvm_registry : CVMRegistry
        Registry containing model data and configurations
    vm1d_data : VelocityModel1D
        1D velocity model data
    nz_tomography_data : TomographyData
        Tomography model data for New Zealand
    global_surfaces : list[GlobalSurfaceRead]
        List of global surface models
    basin_data_list : list[BasinData]
        List of basin model data
    vm_params : dict
        Velocity model parameters
    out_dir : Path
        Output directory for results
    np_workers : int
        Number of worker processes to use
    blas_threads : int
        Number of BLAS threads per worker process
    logger : logging.Logger
        Logger for progress reporting
    """
    from velocity_modelling.write.hdf5 import start_hdf5_writer_process

    total_slices = len(global_mesh.y)
    logger.info(
        f"Processing {total_slices} slices individually using {np_workers} workers"
    )

    # Start timing parallel setup
    parallel_setup_start_time = time.time()

    # Set up global variables for worker processes
    _setup_parallel_globals(
        cvm_registry,
        vm1d_data,
        nz_tomography_data,
        global_surfaces,
        basin_data_list,
        global_mesh,
        basin_membership,
        partial_global_mesh_list,
        vm_params,
    )

    # Configure HDF5 for multiprocessing
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    # Create queue for worker-to-writer communication
    mp_context = mp.get_context("spawn")
    queue = mp_context.Queue(maxsize=np_workers * 2)

    # Start HDF5 writer process
    writer_process = start_hdf5_writer_process(queue, str(out_dir), vm_params, logger)

    try:
        # Submit individual slices to worker processes
        with ProcessPoolExecutor(
            max_workers=np_workers,
            mp_context=mp.get_context("fork"),
            initializer=_init_worker_process,
            initargs=(queue, blas_threads),
        ) as executor:
            # Submit all slices for processing
            futures = {
                executor.submit(_compute_slice, j): j for j in range(total_slices)
            }

            # Process completed slices with progress tracking
            completed_slices = 0
            first_slice_completed = False

            with tqdm(
                total=total_slices, desc="Processing slices", unit="slice"
            ) as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()  # Raise any exceptions from workers
                        completed_slices += 1
                        pbar.update(1)

                        # Log parallel setup time after first slice completes
                        if not first_slice_completed:
                            parallel_setup_elapsed_time = (
                                time.time() - parallel_setup_start_time
                            )
                            logger.info(
                                f"Getting ready for parallel computation in {parallel_setup_elapsed_time:.2f} seconds"
                            )
                            first_slice_completed = True

                    except (RuntimeError, ValueError) as e:
                        logger.error(f"Worker process failed: {e}")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        raise RuntimeError(f"Parallel processing failed: {e}")
                    except (OSError, IOError) as e:
                        logger.error(f"File system error in worker: {e}")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        raise OSError(f"Parallel processing failed: {e}")

    finally:
        # Signal writer process to finish and wait
        queue.put(None)  # Sentinel value
        writer_process.join(timeout=30)
        if writer_process.is_alive():
            logger.warning("Writer process did not terminate gracefully")
            writer_process.terminate()
            writer_process.join()

    logger.info("Parallel processing completed successfully")


# ============================================================================
# Main Processing Function
# ============================================================================


def _generate_velocity_model_impl(
    nzcvm_cfg_path: Path,
    out_dir: Path | None = None,
    nzcvm_registry: Path | None = None,
    model_version: str | None = None,
    output_format_str: str = WriteFormat.EMOD3D.name,
    nzcvm_data_root: Path | None = None,
    smoothing: bool = False,
    np_workers: int = 1,
    blas_threads: int | None = None,
    log_level: str = "INFO",
) -> None:
    """
    Implementation of velocity model generation.

    This function orchestrates the complete velocity model generation process,
    including setup, data loading, and choosing between serial or parallel processing.

    Parameters
    ----------
    nzcvm_cfg_path : Path
        Path to the NZCVM configuration file
    out_dir : Path, optional
        Output directory (overrides config file setting)
    nzcvm_registry : Path, optional
        Path to NZCVM registry file
    model_version : str, optional
        Model version to use (overrides config file setting)
    output_format_str : str
        Output format name (EMOD3D, HDF5, CSV)
    nzcvm_data_root : Path, optional
        Data root directory (overrides config file setting)
    smoothing : bool
        Whether to apply smoothing (not yet implemented)
    np_workers : int
        Number of worker processes. Default is 1 (serial processing).
    blas_threads : int, optional
        Number of BLAS threads per worker process
    log_level : str
        Logging level
    """

    # Configure logging
    logger.setLevel(getattr(logging, log_level.upper()))
    start_time = time.time()

    # Parse output format
    try:
        output_format = WriteFormat[output_format_str]
    except KeyError:
        valid_formats = [fmt.name for fmt in WriteFormat]
        raise ValueError(
            f"Invalid output format '{output_format_str}'. Valid options: {valid_formats}"
        )

    logger.log(
        logging.INFO,
        f"Starting velocity model generation with {output_format.value} output format",
    )

    # Determine processing mode and configure worker/thread allocation
    if np_workers <= 1:
        logger.log(logging.INFO, "Using serial processing")
        use_parallel = False
        actual_workers = 1
        actual_blas_threads = blas_threads or (os.cpu_count() or 2)
    else:
        # Process + BLAS budgeting with core limit enforcement
        cores = os.cpu_count() or 2
        actual_workers = min(
            np_workers, cores
        )  # This limits np_workers to available cores
        actual_blas_threads = blas_threads or max(1, cores // actual_workers)

        # Warn user if they requested more workers than available cores
        if np_workers > cores:
            logger.warning(
                f"Requested {np_workers} workers but only {cores} CPU cores available. "
                f"Using {actual_workers} workers instead."
            )

        logger.log(
            logging.INFO, f"Using parallel processing with {actual_workers} workers"
        )
        logger.log(logging.INFO, f"BLAS threads per worker: {actual_blas_threads}")
        use_parallel = True

    try:
        # Load configuration and setup data paths
        logger.log(logging.INFO, "Loading configuration parameters")
        vm_params = _load_vm_params_from_cfg(
            nzcvm_cfg_path, out_dir, nzcvm_data_root, model_version
        )

        if nzcvm_data_root is not None:
            data_root = nzcvm_data_root
            logger.log(logging.INFO, f"Using CLI-specified data root: {data_root}")
        else:
            data_root = get_data_root()
            logger.log(logging.INFO, f"Using default data root: {data_root}")

        if nzcvm_registry is None:
            from velocity_modelling.constants import get_registry_path

            registry_path = get_registry_path(data_root=data_root)
        else:
            registry_path = nzcvm_registry

        logger.log(logging.INFO, f"Using model version: {vm_params['model_version']}")
        logger.log(logging.INFO, f"Using data root: {data_root}")
        logger.log(logging.INFO, f"Using registry: {registry_path}")

        # Initialize CVM registry
        cvm_registry = CVMRegistry(
            version=vm_params["model_version"],
            data_root=data_root,
            registry_path=registry_path,
            logger=logger,
        )

        # Add this right after the CVMRegistry initialization:
        logger.log(
            logging.INFO, f"After CVMRegistry init, data_root still = {data_root}"
        )

        # Setup mesh and model data
        logger.log(logging.INFO, "Generating global mesh")
        global_mesh = gen_full_model_grid_great_circle(vm_params, logger)

        logger.log(logging.INFO, "Loading model data")
        vm1d_data, nz_tomography_data, global_surfaces, basin_data_list = (
            cvm_registry.load_all_global_data()
        )

        # Preprocess basin membership for efficiency
        logger.log(logging.INFO, "Preprocessing basin membership")

        basin_membership, partial_global_mesh_list = BasinMembership.from_dense_grid(
            global_mesh,
            basin_data_list,
            smooth_boundary=nz_tomography_data.smooth_boundary,
            np_workers=actual_workers,
            logger=logger,
        )

        # Set final output directory
        final_out_dir = out_dir or vm_params["output_dir"]
        final_out_dir.mkdir(parents=True, exist_ok=True)
        logger.log(logging.INFO, f"Output directory: {final_out_dir}")

        # Determine output writer function based on format
        if output_format == WriteFormat.HDF5:
            from velocity_modelling.write.hdf5 import write_global_qualities

            logger.log(logging.INFO, "Using HDF5 output format")
        elif output_format == WriteFormat.EMOD3D:
            from velocity_modelling.write.emod3d import write_global_qualities

            logger.log(logging.INFO, "Using EMOD3D output format")
        elif output_format == WriteFormat.CSV:
            from velocity_modelling.write.csv import write_global_qualities

            logger.log(logging.INFO, "Using CSV output format")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Choose processing mode
        if not use_parallel:
            # Serial processing
            _run_serial_processing(
                global_mesh,
                basin_membership,
                partial_global_mesh_list,
                cvm_registry,
                vm1d_data,
                nz_tomography_data,
                global_surfaces,
                basin_data_list,
                vm_params,
                final_out_dir,
                write_global_qualities,
                smoothing,
                logger,
            )
        else:
            # Parallel processing - enforce HDF5 output format
            if output_format != WriteFormat.HDF5:
                logger.warning(
                    f"Parallel processing requires HDF5 output format, changing from {output_format.name} to HDF5"
                )
                output_format = WriteFormat.HDF5
                # Re-import the correct writer function
                from velocity_modelling.write.hdf5 import write_global_qualities

                logger.log(
                    logging.INFO,
                    "Using HDF5 output format (enforced for parallel processing)",
                )

            _run_parallel_processing(
                global_mesh,
                basin_membership,
                partial_global_mesh_list,
                cvm_registry,
                vm1d_data,
                nz_tomography_data,
                global_surfaces,
                basin_data_list,
                vm_params,
                final_out_dir,
                actual_workers,
                actual_blas_threads,
                logger,
            )

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        raise
    except (ValueError, KeyError) as e:
        logger.log(logging.ERROR, f"Configuration or data error: {e}")
        raise
    except FileNotFoundError as e:
        logger.log(logging.ERROR, f"Required file not found: {e}")
        raise
    except (OSError, IOError) as e:
        logger.log(logging.ERROR, f"File system error: {e}")
        raise
    except RuntimeError as e:
        logger.log(logging.ERROR, f"Processing error: {e}")
        raise

    # Report completion
    elapsed_time = time.time() - start_time
    logger.log(
        logging.INFO,
        f"Velocity model generation completed successfully in {elapsed_time:.1f} seconds",
    )


# ============================================================================
# CLI Entry Point
# ============================================================================


@cli.from_docstring(app)
def generate_3d_model(
    nzcvm_cfg_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out_dir: Annotated[Path | None, typer.Option(file_okay=False)] = None,
    nzcvm_registry: Annotated[
        Path | None,
        typer.Option(
            exists=False,
            dir_okay=False,
            help="Path to nzcvm_registry.yaml (default: nzcvm_data/nzcvm_registry.yaml)",
        ),
    ] = None,
    model_version: str | None = None,
    output_format: str = WriteFormat.EMOD3D.name,
    nzcvm_data_root: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            exists=False,  # will validate later
        ),
    ] = None,
    smoothing: bool = False,  # placeholder for smoothing, not implemented yet
    np_workers: Annotated[int, typer.Option("--np")] = 1,
    blas_threads: int | None = None,
    log_level: str = "INFO",
) -> None:
    """
    Generate 3D seismic velocity model from configuration file.

    This command creates a 3D velocity model by combining global velocity models,
    regional tomographic data, and local basin models according to the parameters
    specified in the configuration file.

    Parameters
    ----------
    nzcvm_cfg_path : Path
        Path to the NZCVM configuration file.
    out_dir : Path, optional
        Output directory (overrides config file setting).
    nzcvm_registry : Path, optional
        Path to nzcvm_registry.yaml file.
    model_version : str, optional
        Model version to use (overrides config file setting).
    output_format : str
        Output format (EMOD3D, HDF5, CSV).
    nzcvm_data_root : Path, optional
        Override the default nzcvm_data directory.
    smoothing : bool
        Enable smoothing (placeholder, not implemented).
    np_workers : int
        Number of parallel workers. Default is 1 (serial processing).
    blas_threads : int, optional
        BLAS threads per worker. If None, np_workers==1 (serial) uses all cores, else
        cores//np_workers (minimum 1).
    log_level : str
        Logging level.

    Examples
    --------
    Basic usage:
        nzcvm generate-3d-model config.cfg

    Parallel processing with custom output:
        nzcvm generate-3d-model config.cfg --np 8 --out-dir ./output

    HDF5 output with specific model version and BLAS threading:
        nzcvm generate-3d-model config.cfg --output-format HDF5 --model-version 2.07 --blas-threads 2
    """

    try:
        _generate_velocity_model_impl(
            nzcvm_cfg_path=nzcvm_cfg_path,
            out_dir=out_dir,
            nzcvm_registry=nzcvm_registry,
            model_version=model_version,
            output_format_str=output_format,
            nzcvm_data_root=nzcvm_data_root,
            smoothing=smoothing,
            np_workers=np_workers,
            blas_threads=blas_threads,
            log_level=log_level,
        )
    except KeyboardInterrupt:
        logger.log(logging.INFO, "Operation cancelled by user")
        raise typer.Exit(1)
    except (ValueError, KeyError) as e:
        logger.log(logging.ERROR, f"Configuration error: {e}")
        raise typer.Exit(1)
    except FileNotFoundError as e:
        logger.log(logging.ERROR, f"File not found: {e}")
        raise typer.Exit(1)
    except (OSError, IOError) as e:
        logger.log(logging.ERROR, f"File system error: {e}")
        raise typer.Exit(1)
    except RuntimeError as e:
        logger.log(logging.ERROR, f"Processing failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
