"""
Module for writing velocity model data in HDF5 format.

This module provides functions to write the velocity model data to HDF5 format files.
HDF5 (Hierarchical Data Format version 5) is an efficient binary format for storing large
scientific datasets with metadata.

The module supports both serial and parallel writing modes, with a dedicated writer process
for parallel processing to ensure thread-safe HDF5 operations.
"""

import atexit
import datetime
import logging
import multiprocessing as mp
import os
import time
from logging import Logger
from pathlib import Path

import h5py
import numpy as np

from velocity_modelling.geometry import PartialGlobalMesh
from velocity_modelling.velocity3d import PartialGlobalQualities

vm_file_name = "velocity_model.h5"

# ============================================================================
# XDMF File Creation
# ============================================================================


def create_xdmf_file(hdf5_file: Path, vm_params: dict, logger: logging.Logger) -> None:
    """
    Create an XDMF file to make the HDF5 file compatible with ParaView.

    Parameters
    ----------
    hdf5_file : Path
        Path to the HDF5 file
    vm_params : dict
        Dictionary containing model parameters
    logger : logging.Logger
        Logger for reporting progress and errors

    Raises
    ------
    KeyError
        If required parameters (nx, ny, nz) are missing from vm_params
    RuntimeError
        If XDMF file creation fails
    """
    xdmf_file = hdf5_file.with_suffix(".xdmf")
    nx = vm_params.get("nx")
    ny = vm_params.get("ny")
    nz = vm_params.get("nz")

    if not all([nx, ny, nz]):
        error_msg = "Missing 'nx' 'ny' or 'nz' key in vm_params. Ensure the velocity model parameters are correctly set."
        logger.log(logging.ERROR, error_msg)
        raise KeyError("Missing nx, ny, or nz in vm_params")

    hdf5_relative = hdf5_file.name
    xdmf_content = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="Velocity_Model" GridType="Uniform">
      <Topology TopologyType="3DRectMesh" Dimensions="{nx} {ny} {nz}"/>
      <Geometry GeometryType="VXVYVZ">
        <DataItem Dimensions="{nx}" NumberType="Int" Precision="4" Format="HDF">
          {hdf5_relative}:/mesh/x
        </DataItem>
        <DataItem Dimensions="{ny}" NumberType="Int" Precision="4" Format="HDF">
          {hdf5_relative}:/mesh/y
        </DataItem>
        <DataItem Dimensions="{nz}" NumberType="Int" Precision="4" Format="HDF">
          {hdf5_relative}:/mesh/z
        </DataItem>
      </Geometry>
      <Attribute Name="P-wave Velocity" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_relative}:/properties/vp
        </DataItem>
      </Attribute>
      <Attribute Name="S-wave Velocity" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_relative}:/properties/vs
        </DataItem>
      </Attribute>
      <Attribute Name="Density" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_relative}:/properties/rho
        </DataItem>
      </Attribute>
      <Attribute Name="Basin Membership" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{nz} {ny} {nx}" NumberType="Int" Precision="1" Format="HDF">
          {hdf5_relative}:/properties/inbasin
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""

    try:
        with open(xdmf_file, "w") as f:
            f.write(xdmf_content)
        logger.log(logging.INFO, f"Created ParaView-compatible XDMF file: {xdmf_file}")
    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            raise
        logger.log(logging.ERROR, f"Error creating XDMF file: {e}")
        raise RuntimeError(f"Failed to create XDMF file: {e}")


# ============================================================================
# HDF5 File Management and Caching
# ============================================================================

# Cache for one open file per out_dir (single-writer process)
_FILE_CACHE = {}  # key: resolved out_dir -> (h5py.File, dict_dsets)

# HDF5 chunk cache configuration for optimal performance
_RDCC_NBYTES = 128 * 1024 * 1024  # 128MB file-level raw chunk cache
_RDCC_NSLOTS = 1_000_003  # large prime, reduces hash collisions
_RDCC_W0 = 0.75  # preemption policy


def _open_file_with_retry(
    hdf5_file: Path, rdcc_kwargs: dict, max_retries: int = 10
) -> h5py.File:
    """
    Open HDF5 file with retry logic for HPC metadata latency.

    Parameters
    ----------
    hdf5_file : Path
        Path to the HDF5 file
    rdcc_kwargs : dict
        Raw data chunk cache configuration parameters
    max_retries : int, optional
        Maximum number of retry attempts (default: 10)

    Returns
    -------
    h5py.File
        Opened HDF5 file handle

    Raises
    ------
    OSError
        If file cannot be opened after all retry attempts
    """
    for attempt in range(max_retries):
        mode = "r+" if hdf5_file.exists() else "w"
        try:
            return h5py.File(hdf5_file, mode, libver="latest", **rdcc_kwargs)
        except OSError as e:
            # ENOENT or transient open error → sleep and retry
            if attempt < max_retries - 1:
                time.sleep(0.05 * (attempt + 1))
            else:
                # Final attempt - let it raise the exception
                raise e


def _create_hdf5_structure(
    f: h5py.File, vm_params: dict, nx: int, ny: int, nz: int
) -> dict:
    """
    Create the HDF5 file structure with groups and datasets.

    If the file already exists and has the structure, reuse it.
    Otherwise create new structure.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle
    vm_params : dict
        Velocity model parameters
    nx : int
        Number of points in x direction
    ny : int
        Number of points in y direction
    nz : int
        Number of points in z direction

    Returns
    -------
    dict
        Dictionary of created datasets for easy access
    """
    # Check if structure already exists and has correct dimensions
    if (
        "config" in f
        and "mesh" in f
        and "properties" in f
        and "vp" in f["properties"]
        and f["properties"]["vp"].shape == (nz, ny, nx)
    ):
        # File structure already exists with correct dimensions, reuse it
        mesh_group = f["mesh"]
        props = f["properties"]

        # Update file-level attributes in case they changed
        f.attrs.update(
            {
                "total_y_slices": ny,
                "format_version": "1.0",
                "creation_date": datetime.datetime.now().isoformat(),
            }
        )

        return {
            "vp": props["vp"],
            "vs": props["vs"],
            "rho": props["rho"],
            "inbasin": props["inbasin"],
            "lat": mesh_group["lat"],
            "lon": mesh_group["lon"],
        }

    # File doesn't exist or has wrong dimensions - recreate structure
    # Clear any existing groups first
    for group_name in ["config", "mesh", "properties"]:
        if group_name in f:
            del f[group_name]

    # Set file-level attributes
    f.attrs.update(
        {
            "total_y_slices": ny,
            "format_version": "1.0",
            "creation_date": datetime.datetime.now().isoformat(),
        }
    )

    # Create configuration group with stringified parameter values
    config_group = f.create_group("config")
    config_group.attrs.update(
        {
            k: (
                v.name
                if hasattr(v, "name")
                else v.value
                if hasattr(v, "value")
                else str(v)
            )
            for k, v in vm_params.items()
        }
    )
    config_group.attrs["config_string"] = "\n".join(
        f"{k.upper()}={v}" for k, v in vm_params.items()
    )

    # Create mesh group with coordinate arrays
    mesh_group = f.create_group("mesh")
    mesh_group.create_dataset("x", data=np.arange(nx, dtype=np.int32))
    mesh_group.create_dataset("y", data=np.arange(ny, dtype=np.int32))
    mesh_group.create_dataset("z", data=np.arange(nz, dtype=np.int32))
    d_lat = mesh_group.create_dataset("lat", shape=(nx, ny), dtype="f8")
    d_lon = mesh_group.create_dataset("lon", shape=(nx, ny), dtype="f8")

    # Create properties group with optimized chunking
    props = f.create_group("properties")

    # Use (nz, ny, nx) layout for ParaView compatibility
    # Chunk by complete y-slices (nz, 1, nx) for fast slice writes
    shape = (nz, ny, nx)
    chunks = (nz, 1, nx)  # One complete y-slice per chunk

    d_vp = props.create_dataset("vp", shape=shape, dtype="f4", chunks=chunks)
    d_vs = props.create_dataset("vs", shape=shape, dtype="f4", chunks=chunks)
    d_rho = props.create_dataset("rho", shape=shape, dtype="f4", chunks=chunks)
    d_inb = props.create_dataset("inbasin", shape=shape, dtype="i1", chunks=chunks)

    # Add dataset attributes
    d_vp.attrs["units"] = "km/s"
    d_vs.attrs.update(
        {"units": "km/s", "min_value_enforced": vm_params.get("min_vs", 0.0)}
    )
    d_rho.attrs["units"] = "g/cm^3"

    return {
        "vp": d_vp,
        "vs": d_vs,
        "rho": d_rho,
        "inbasin": d_inb,
        "lat": d_lat,
        "lon": d_lon,
    }


def _ensure_hdf5_file_open(
    out_dir: Path,
    vm_params: dict,
    nx: int,
    ny: int,
    nz: int,
    logger: Logger | None = None,
) -> tuple:
    """
    Ensure HDF5 file is open and return file handle and datasets.

    Opens and initializes the HDF5 file once, then caches it for subsequent use.
    This function is safe to call multiple times - it will reuse existing handles.

    Parameters
    ----------
    out_dir : Path
        Output directory path
    vm_params : dict
        Velocity model parameters
    nx : int
        Number of points in x direction
    ny : int
        Number of points in y direction
    nz : int
        Number of points in z direction
    logger : Logger, optional
        Logger for debug messages

    Returns
    -------
    tuple
        (h5py.File, dict) - File handle and dataset dictionary
    """
    # Use resolved path as cache key to handle symbolic links correctly
    key = str(Path(out_dir).resolve())
    cached = _FILE_CACHE.get(key)
    if cached:
        return cached

    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_file = out_dir / vm_file_name

    # Configure chunk cache for optimal performance
    rdcc_kwargs = {
        "rdcc_nbytes": _RDCC_NBYTES,
        "rdcc_nslots": _RDCC_NSLOTS,
        "rdcc_w0": _RDCC_W0,
    }

    # Open file with retry logic for HPC environments
    f = _open_file_with_retry(hdf5_file, rdcc_kwargs)

    # Create HDF5 structure if this is a new file
    dsets = _create_hdf5_structure(f, vm_params, nx, ny, nz)

    # Cache the file handle and datasets
    _FILE_CACHE[key] = (f, dsets)

    # Register cleanup function to close file on process exit
    def _cleanup_file(cache_key: str = key):
        """
        Clean up cached file handle on process exit.

        Parameters
        ----------
        cache_key : str
            Cache key for the file to close
        """
        cached_data = _FILE_CACHE.pop(cache_key, None)
        if cached_data:
            try:
                cached_data[0].close()
            except (OSError, ValueError):
                pass  # Ignore errors during cleanup

    # Register the cleanup function only once
    atexit.register(_cleanup_file)

    if logger:
        # Define shape and chunks for logging
        shape = (nz, ny, nx)
        chunks = (nz, 1, nx)
        logger.log(
            logging.DEBUG,
            f"[hdf5] Opened {hdf5_file} with shape={shape} chunks={chunks} "
            f"cache={_RDCC_NBYTES // (1024 * 1024)}MB",
        )

    return _FILE_CACHE[key]


def close_hdf5_cache(out_dir: Path | None = None) -> None:
    """
    Manually close cached HDF5 file handles.

    This function is useful for testing or when you need to ensure files
    are closed before the process exits.

    Parameters
    ----------
    out_dir : Path, optional
        If specified, close only the file for this directory.
        If None, close all cached files.
    """
    if out_dir is None:
        keys = list(_FILE_CACHE.keys())
    else:
        keys = [str(Path(out_dir).resolve())]

    for key in keys:
        cached_data = _FILE_CACHE.pop(key, None)
        if cached_data:
            try:
                cached_data[0].close()
            except (OSError, ValueError):
                pass  # Ignore errors during cleanup


# ============================================================================
# Core Writing Functions
# ============================================================================


def write_global_qualities(
    out_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities: PartialGlobalQualities,
    lat_ind: int,
    vm_params: dict,
    logger: Logger | None = None,
) -> None:
    """
    Write a latitude slice of velocity data to the consolidated HDF5 file.

    This function writes velocity model data for a single y-slice to the HDF5 file.
    It handles both file creation (for the first slice) and subsequent slice updates
    in an efficient, thread-safe manner.

    Parameters
    ----------
    out_dir : Path
        Output directory where the HDF5 file will be written
    partial_global_mesh : PartialGlobalMesh
        Mesh data for the current latitude slice
    partial_global_qualities : PartialGlobalQualities
        Velocity and density data for the current latitude slice
    lat_ind : int
        Latitude index (y-coordinate) for this slice
    vm_params : dict
        Velocity model parameters from configuration
    logger : Logger, optional
        Logger instance for progress messages

    Raises
    ------
    KeyError
        If 'ny' parameter is missing from vm_params
    ValueError
        If data shapes don't match expected dimensions,
        or if any velocity/density values are negative
    OSError
        If HDF5 file operations fail
    """
    if logger is None:
        logger = logging.getLogger("hdf5")

    vs_data = np.copy(partial_global_qualities.vs)

    # Apply minimum Vs constraint before writing
    min_vs = vm_params.get("min_vs", 0.0)

    # Validate that all velocity/density values are non-negative
    vp_data = partial_global_qualities.vp
    rho_data = partial_global_qualities.rho

    if np.any(vp_data < 0):
        error_msg = f"Negative values found in Vp data at slice {lat_ind}. Min value: {np.min(vp_data)}"
        logger.log(logging.ERROR, error_msg)
        raise ValueError(error_msg)

    if np.any(vs_data < 0):
        error_msg = f"Negative values found in Vs data at slice {lat_ind}. Min value: {np.min(vs_data)}"
        logger.log(logging.ERROR, error_msg)
        raise ValueError(error_msg)

    if np.any(rho_data < 0):
        error_msg = f"Negative values found in density data at slice {lat_ind}. Min value: {np.min(rho_data)}"
        logger.log(logging.ERROR, error_msg)
        raise ValueError(error_msg)

    vs_data[vs_data < min_vs] = min_vs

    # Extract dimensions
    nx, nz = partial_global_qualities.vp.shape
    try:
        ny = int(vm_params["ny"])
    except KeyError:
        logger.log(logging.ERROR, "Missing 'ny' parameter in vm_params")
        raise KeyError("Missing 'ny' key in vm_params")

    # Get or create HDF5 file handle and datasets
    f, dsets = _ensure_hdf5_file_open(out_dir, vm_params, nx, ny, nz, logger)

    # Write mesh coordinates for this y-slice
    dsets["lat"][:, lat_ind] = partial_global_mesh.lat
    dsets["lon"][:, lat_ind] = partial_global_mesh.lon

    # Transpose data from (nx, nz) to (nz, nx) for HDF5 layout
    # HDF5 uses (nz, ny, nx) order for ParaView compatibility
    vp_transposed = partial_global_qualities.vp.T
    vs_transposed = vs_data.T
    rho_transposed = partial_global_qualities.rho.T
    inbasin_transposed = partial_global_qualities.inbasin.T

    # Validate data shape
    expected_shape = (nz, nx)
    if vp_transposed.shape != expected_shape:
        error_msg = (
            f"Shape mismatch: got {vp_transposed.shape}, expected {expected_shape}"
        )
        logger.log(logging.ERROR, error_msg)
        raise ValueError(error_msg)

    # Write property data for this y-slice using optimized slice notation
    dsets["vp"][:, lat_ind, :] = vp_transposed
    dsets["vs"][:, lat_ind, :] = vs_transposed
    dsets["rho"][:, lat_ind, :] = rho_transposed
    dsets["inbasin"][:, lat_ind, :] = inbasin_transposed

    # Update progress tracking attribute
    f.attrs["last_slice_written"] = lat_ind

    logger.log(logging.DEBUG, f"Successfully wrote slice {lat_ind} to HDF5 file")


# ============================================================================
# Parallel Processing Support
# ============================================================================


def _hdf5_writer_process(
    queue: mp.Queue,
    out_dir_str: str,
    vm_params: dict,
    logger: logging.Logger,
) -> None:
    """
    Dedicated writer process for parallel HDF5 operations.

    This process runs in a separate Python process and handles all HDF5 write
    operations to ensure thread safety. It receives slice data from worker
    processes via a queue and writes them to the HDF5 file sequentially.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue for receiving slice data from worker processes
    out_dir_str : str
        Output directory path as string (for multiprocessing compatibility)
    vm_params : dict
        Velocity model parameters
    logger : logging.Logger
        Logger instance from the main process

    Notes
    -----
    This function runs in a separate process started with 'spawn' context
    to ensure clean HDF5 library initialization. It processes incoming
    slice data until it receives a sentinel None value.
    """
    import logging
    from pathlib import Path

    # Configure HDF5 for multiprocessing safety
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    logger.log(logging.DEBUG, "HDF5 writer process started, waiting for slice data...")

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    received_slices = 0
    xdmf_created = False  # Flag to prevent duplicate XDMF creation

    try:
        # Process incoming slice data until sentinel received
        while True:
            item = queue.get()
            if item is None:  # Sentinel value to stop processing
                logger.log(
                    logging.DEBUG,
                    f"Writer process received sentinel, processed {received_slices} slices",
                )
                break

            if received_slices == 0:
                logger.log(logging.DEBUG, "Received first slice data")

            # Unpack slice data and write to HDF5
            j, partial_mesh, partial_qualities = item
            write_global_qualities(
                out_dir, partial_mesh, partial_qualities, j, vm_params, logger
            )
            received_slices += 1

        # Create ParaView-compatible XDMF file after all slices are written
        # Only create once to prevent duplicates
        if not xdmf_created:
            xdmf_created = True  # Set flag immediately to prevent duplicate execution
            try:
                hdf5_file = out_dir / vm_file_name
                create_xdmf_file(hdf5_file, vm_params, logger)
                logger.log(
                    logging.INFO, "HDF5 model completed with ParaView compatibility"
                )
                xdmf_created = True
            except (OSError, KeyError, RuntimeError) as e:
                logger.log(logging.WARNING, f"Failed to create XDMF file: {e}")
                xdmf_created = False  # Reset flag if creation failed

    except Exception as e:
        logger.log(logging.ERROR, f"HDF5 writer process error: {e}")
        raise
    finally:
        # Clean up cached file handles
        try:
            close_hdf5_cache(out_dir)
        except (OSError, ValueError) as e:
            logger.log(logging.WARNING, f"Failed to close HDF5 cache: {e}")

    logger.log(logging.DEBUG, "HDF5 writer process finished")


def start_hdf5_writer_process(
    queue: mp.Queue,
    out_dir_str: str,
    vm_params: dict,
    logger: logging.Logger,
) -> mp.Process:
    """
    Start the dedicated HDF5 writer process for parallel processing.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue for sending slice data to the writer process
    out_dir_str : str
        Output directory path as string (for multiprocessing compatibility)
    vm_params : dict
        Velocity model parameters
    logger : logging.Logger
        Logger instance from the main process

    Returns
    -------
    multiprocessing.Process
        Started writer process instance
    """
    # Use 'spawn' context for clean HDF5 initialization
    mp_context = mp.get_context("spawn")
    writer_process = mp_context.Process(
        target=_hdf5_writer_process,
        args=(queue, out_dir_str, vm_params, logger),
        name="HDF5Writer",
    )
    writer_process.start()
    return writer_process
