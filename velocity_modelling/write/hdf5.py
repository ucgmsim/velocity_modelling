"""
Module for writing velocity model data in HDF5 format.

This module provides functions to write the velocity model data to HDF5 format files.
HDF5 (Hierarchical Data Format version 5) is an efficient binary format for storing large
scientific datasets with metadata.
"""
import datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from velocity_modelling.geometry import PartialGlobalMesh
from velocity_modelling.velocity3d import PartialGlobalQualities


def create_xdmf_file(hdf5_file: Path, vm_params: dict, logger: logging.Logger) -> None:
    """
    Create an XDMF file to make the HDF5 file compatible with ParaView.

    Parameters
    ----------
    hdf5_file : Path
        Path to the HDF5 file.
    vm_params : dict
        Dictionary containing model parameters.
    logger : logging.Logger
        Logger for reporting progress and errors.
    """
    xdmf_file = hdf5_file.with_suffix(".xdmf")
    nx = vm_params.get("nx")
    ny = vm_params.get("ny")
    nz = vm_params.get("nz")
    if not all([nx, ny, nz]):
        logger.log(
            logging.ERROR,
            "Missing 'nx' 'ny' or 'nz' key in vm_params. Ensure the velocity model parameters are correctly set.",
        )
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

# --- Cache for one open file per out_dir (single-writer process) ---
_FILE_CACHE = {}  # key: resolved out_dir -> (h5py.File, dict_dsets)
_RDCC_NBYTES = 128 * 1024 * 1024  # 128MB file-level raw chunk cache
_RDCC_NSLOTS = 1_000_003          # large prime, reduces hash collisions
_RDCC_W0     = 0.75               # preemption policy

def _ensure_open(out_dir: Path, vm_params: dict, nx: int, ny: int, nz: int, logger: Logger):
    """
    Open/create velocity_model.h5 once and return (f, dsets).
    dsets: {"vp":ds, "vs":ds, "rho":ds, "inbasin":ds, "lat":ds, "lon":ds}
    """
    from pathlib import Path as _P
    key = str(_P(out_dir).resolve())
    cached = _FILE_CACHE.get(key)
    if cached:
        return cached

    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_file = out_dir / "velocity_model.h5"

    # Open once with tuned raw data chunk cache
    f = h5py.File(
        hdf5_file, "w", libver="latest",
        rdcc_nbytes=_RDCC_NBYTES, rdcc_nslots=_RDCC_NSLOTS, rdcc_w0=_RDCC_W0
    )

    # File-level attrs
    f.attrs.update(
        {
            "total_y_slices": ny,
            "format_version": "1.0",
            "creation_date": datetime.datetime.now().isoformat(),
        }
    )

    # Config group (stringify exotic values just like your current code)
    config_group = f.create_group("config")
    config_group.attrs.update(
        {
            k: (v.name if hasattr(v, "name") else v.value if hasattr(v, "value") else str(v))
            for k, v in vm_params.items()
        }
    )
    config_group.attrs["config_string"] = "\n".join(f"{k.upper()}={v}" for k, v in vm_params.items())

    # Mesh + properties groups
    mesh_group = f.create_group("mesh")
    mesh_group.create_dataset("x", data=np.arange(nx, dtype=np.int32))
    mesh_group.create_dataset("y", data=np.arange(ny, dtype=np.int32))
    mesh_group.create_dataset("z", data=np.arange(nz, dtype=np.int32))
    d_lat = mesh_group.create_dataset("lat", shape=(nx, ny), dtype="f8")
    d_lon = mesh_group.create_dataset("lon", shape=(nx, ny), dtype="f8")

    props = f.create_group("properties")

    # Your layout is (nz, ny, nx) (z, y, x) for ParaView; write whole y-slice fast:
    shape  = (nz, ny, nx)
    chunks = (nz, 1,  nx)  # one complete y-slice per chunk  ← key for fast [:, j, :] writes

    d_vp  = props.create_dataset("vp",      shape=shape, dtype="f4", chunks=chunks)
    d_vs  = props.create_dataset("vs",      shape=shape, dtype="f4", chunks=chunks)
    d_rho = props.create_dataset("rho",     shape=shape, dtype="f4", chunks=chunks)
    d_inb = props.create_dataset("inbasin", shape=shape, dtype="i1", chunks=chunks)
    d_vp.attrs["units"] = "km/s"
    d_vs.attrs.update({"units": "km/s", "min_value_enforced": vm_params.get("min_vs", 0.0)})
    d_rho.attrs["units"] = "g/cm^3"

    dsets = {"vp": d_vp, "vs": d_vs, "rho": d_rho, "inbasin": d_inb, "lat": d_lat, "lon": d_lon}
    _FILE_CACHE[key] = (f, dsets)

    # Close automatically when the process exits
    import atexit
    def _close_once(k=key):
        tup = _FILE_CACHE.pop(k, None)
        if tup:
            try: tup[0].close()
            except Exception: pass
    atexit.register(_close_once)

    if logger:
        logger.log(
            logging.DEBUG,
            f"[hdf5] opened {hdf5_file} shape={shape} chunks={chunks} rdcc={_RDCC_NBYTES//(1024*1024)}MB",
        )
    return _FILE_CACHE[key]

def close_cache(out_dir: Optional[Path] = None):
    """Optional manual close (useful in tests/resets)."""
    if out_dir is None:
        keys = list(_FILE_CACHE.keys())
    else:
        keys = [str(Path(out_dir).resolve())]
    for k in keys:
        tup = _FILE_CACHE.pop(k, None)
        if tup:
            try: tup[0].close()
            except Exception: pass


# def write_global_qualities(
#     out_dir: Path,
#     partial_global_mesh: PartialGlobalMesh,
#     partial_global_qualities: PartialGlobalQualities,
#     lat_ind: int,
#     vm_params: dict,
#     logger: Optional[Logger] = None,
# ) -> None:
#     """
#     Write a latitude slice of velocity data to a single consolidated HDF5 file.
#
#     For the first slice (lat_ind=0), this creates the file and initializes the structure.
#     For subsequent slices, it adds data to the existing file.
#
#     Parameters
#     ----------
#     out_dir : Path
#         Path to the output directory where the HDF5 file will be written.
#     partial_global_mesh : PartialGlobalMesh
#         Mesh data for the current latitude slice.
#     partial_global_qualities : PartialGlobalQualities
#         Velocity and density data for the current latitude slice.
#     lat_ind : int
#         Latitude index to determine the write mode (write or append).
#     vm_params : dict
#         Dictionary containing velocity model parameters from nzcvm.cfg.
#     logger : Logger, optional
#         Logger instance for logging messages.
#
#     Raises
#     ------
#     OSError
#         If the HDF5 file cannot be written due to permissions or disk issues.
#     """
#     if logger is None:
#         logger = Logger("xdmf")
#
#     # Output filename - single file for all slices
#     hdf5_file = out_dir / "velocity_model.h5"
#     vs_data = np.copy(partial_global_qualities.vs)
#     min_vs = vm_params.get("min_vs", 0.0)
#     vs_data[vs_data < min_vs] = min_vs
#
#     # ny is the number of slices in the y direction, but what we are processing here is a single slice
#     # along y (=lat if not-rotated) axis, so we need to get the total number of slices from the vm_params
#
#     try:
#         ny = vm_params["ny"]
#     except KeyError:
#         logger.log(
#             logging.ERROR,
#             "Missing 'ny' key in vm_params. Ensure the velocity model parameters are correctly set.",
#         )
#         raise KeyError("Missing 'ny' key in vm_params.")
#
#     nx, nz = partial_global_qualities.vp.shape
#
#     # If first slice, create the file and initialize structure
#     if lat_ind == 0:
#         try:
#             with h5py.File(hdf5_file, "w") as f:
#                 f.attrs.update(
#                     {
#                         "total_y_slices": ny,
#                         "format_version": "1.0",
#                         "creation_date": datetime.datetime.now().isoformat(),
#                     }
#                 )
#
#                 config_group = f.create_group("config")
#                 config_group.attrs.update(
#                     {
#                         k: (
#                             v.name
#                             if hasattr(v, "name")
#                             else v.value
#                             if hasattr(v, "value")
#                             else str(v)
#                         )
#                         for k, v in vm_params.items()
#                     }
#                 )
#                 config_group.attrs["config_string"] = "\n".join(
#                     f"{k.upper()}={v}" for k, v in vm_params.items()
#                 )
#
#                 mesh_group = f.create_group("mesh")
#                 mesh_group.create_dataset("x", data=np.arange(nx, dtype=np.int32))
#                 mesh_group.create_dataset("y", data=np.arange(ny, dtype=np.int32))
#                 mesh_group.create_dataset(
#                     "z", data=np.arange(nz, dtype=np.int32)
#                 )  # depth 0 is the top, the higher z is the deeper
#                 mesh_group.create_dataset("lat", shape=(nx, ny), dtype="f8")
#                 mesh_group.create_dataset("lon", shape=(nx, ny), dtype="f8")
#
#                 props = f.create_group("properties")
#
#                 # Create datasets with shape (nz, ny, nx): Paraview demands this order
#                 shape = (nz, ny, nx)
#
#                 logger.log(
#                     logging.DEBUG,
#                     f"Creating datasets with shape (nz={nz}, ny={ny}, nx={nx})",
#                 )
#                 # props.create_dataset("vp", shape=shape, dtype="f4", compression="gzip")
#                 # props.create_dataset("vs", shape=shape, dtype="f4", compression="gzip")
#                 # props.create_dataset("rho", shape=shape, dtype="f4", compression="gzip")
#                 # props.create_dataset(
#                 #     "inbasin", shape=shape, dtype="i1", compression="gzip"
#                 # )
#
#                 props.create_dataset("vp", shape=shape, dtype="f4")
#                 props.create_dataset("vs", shape=shape, dtype="f4")
#                 props.create_dataset("rho", shape=shape, dtype="f4")
#                 props.create_dataset(
#                     "inbasin", shape=shape, dtype="i1")
#
#                 # Add metadata
#                 props["vp"].attrs["units"] = "km/s"
#                 props["vs"].attrs.update(
#                     {"units": "km/s", "min_value_enforced": min_vs}
#                 )
#                 props["rho"].attrs["units"] = "g/cm^3"
#         except OSError as e:
#             logger.log(logging.ERROR, f"Failed to create HDF5 file {hdf5_file}: {e}")
#             raise OSError(f"Failed to create HDF5 file {hdf5_file}: {str(e)}")
#
#     try:
#         with h5py.File(hdf5_file, "r+") as f:
#             f["mesh/lat"][:, lat_ind] = partial_global_mesh.lat
#             f["mesh/lon"][:, lat_ind] = partial_global_mesh.lon
#
#             # Write property data for this slice to make this Paraview compatible
#             # partial_global_qualities.vp is (nx, nz), need to transpose to (nz, nx) for (z, x)
#             # Dataset is (nz, ny, nx), so [:, lat_ind, :] expects (nz, nx)
#             vp = partial_global_qualities.vp.T
#             vs = vs_data.T
#             rho = partial_global_qualities.rho.T
#             inbasin = partial_global_qualities.inbasin.T
#
#             expected = (nz, nx)
#             if vp.shape != expected:
#                 logger.log(
#                     logging.ERROR,
#                     f"Shape mismatch: vp has shape {vp.shape}, expected {expected}",
#                 )
#                 raise ValueError(f"Shape mismatch: got {vp.shape}, expected {expected}")
#
#             f["properties/vp"][:, lat_ind, :] = vp
#             f["properties/vs"][:, lat_ind, :] = vs
#             f["properties/rho"][:, lat_ind, :] = rho
#             f["properties/inbasin"][:, lat_ind, :] = inbasin
#
#             f.attrs["last_slice_written"] = lat_ind
#     except OSError as e:
#         logger.log(logging.ERROR, f"Failed to update HDF5 file {hdf5_file}: {e}")
#         raise OSError(f"Failed to update HDF5 file {hdf5_file}: {str(e)}")
#     except Exception as e:
#         if isinstance(e, (SystemExit, KeyboardInterrupt)):
#             raise  # Re-raise critical exceptions
#         logger.log(logging.ERROR, f"Error writing to HDF5 file {hdf5_file}: {e}")
#         raise RuntimeError(f"Error writing to HDF5 file {hdf5_file}: {str(e)}")
#
#     # After writing all slices, create the XDMF file
#     if lat_ind == ny - 1:
#         create_xdmf_file(hdf5_file, vm_params, logger)
#         logger.log(logging.INFO, "HDF5 model complete with ParaView compatibility")


def write_global_qualities(
    out_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities: PartialGlobalQualities,
    lat_ind: int,
    vm_params: dict,
    logger: Optional[Logger] = None,
) -> None:
    if logger is None:
        logger = logging.getLogger("hdf5")

    # Enforce min_vs before writing
    vs_data = np.copy(partial_global_qualities.vs)
    min_vs = vm_params.get("min_vs", 0.0)
    vs_data[vs_data < min_vs] = min_vs

    # nx, ny, nz
    nx, nz = partial_global_qualities.vp.shape
    try:
        ny = int(vm_params["ny"])
    except KeyError:
        logger.log(logging.ERROR, "Missing 'ny' in vm_params.")
        raise

    # Open/create once; reuse thereafter
    f, dsets = _ensure_open(out_dir, vm_params, nx, ny, nz, logger)

    # Write mesh lat/lon for this y-slice
    dsets["lat"][:, lat_ind] = partial_global_mesh.lat
    dsets["lon"][:, lat_ind] = partial_global_mesh.lon

    # (nx, nz) -> (nz, nx) for [:, j, :]
    vp  = partial_global_qualities.vp.T
    vs  = vs_data.T
    rho = partial_global_qualities.rho.T
    inb = partial_global_qualities.inbasin.T

    expected = (nz, nx)
    if vp.shape != expected:
        logger.log(logging.ERROR, f"Shape mismatch: got {vp.shape}, expected {expected}")
        raise ValueError(f"Shape mismatch: got {vp.shape}, expected {expected}")

    dsets["vp"][:,  lat_ind, :]  = vp
    dsets["vs"][:,  lat_ind, :]  = vs
    dsets["rho"][:, lat_ind, :]  = rho
    dsets["inbasin"][:, lat_ind, :] = inb

    # Update progress attr (cheap)
    f.attrs["last_slice_written"] = lat_ind

    # Create XDMF when the last slice *for y* has been written (kept from your version)
    # NOTE: if slices can arrive out-of-order, consider calling create_xdmf_file(...) after the run.
    # if lat_ind == ny - 1:
    #     hdf5_file = Path(out_dir) / "velocity_model.h5"
    #     create_xdmf_file(hdf5_file, vm_params, logger)
    #     logger.log(logging.INFO, "HDF5 model complete with ParaView compatibility")
