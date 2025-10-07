"""
Module for writing a velocity model to file for EMOD3D.

"""

import logging
import struct
import sys
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np

from velocity_modelling.geometry import (
    PartialGlobalMesh,
)
from velocity_modelling.velocity3d import (
    PartialGlobalQualities,
)


def write_global_qualities(
    output_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities: PartialGlobalQualities,
    lat_ind: int,
    vm_params: dict,
    logger: Optional[Logger] = None,
):
    """
    Write the full velocity model to file for EMOD3D

    Parameters
    ----------
    output_dir : Path
        Directory where the output files will be written.
    partial_global_mesh : PartialGlobalMesh
        Structure containing the latitude and longitude grid.
    partial_global_qualities : PartialGlobalQualities
        Structure containing Vp, Vs, and Rho for all grid points.
    lat_ind : int
        Latitude index to determine the write mode (write or append).
    vm_params : dict
        Dictionary containing velocity model parameters from nzcvm.cfg.
    logger : Logger, optional
        Logger instance for logging messages.

    Raises
    ------
    ValueError
        If any velocity/density values are negative.
    OSError
        If file operations fail.
    """
    if logger is None:
        logger = Logger("emod3d.wrote_global_qualities")

    min_vs = vm_params.get(
        "min_vs", 0.0
    )  # Get the minimum Vs value from the parameters

    # Validate that all velocity/density values are non-negative
    vp_data = partial_global_qualities.vp
    vs_data = partial_global_qualities.vs
    rho_data = partial_global_qualities.rho
    inbasin_data = partial_global_qualities.inbasin

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

    # perform endian check
    endianness = sys.byteorder

    endian_format = "<" if endianness == "little" else ">"

    output_dir.mkdir(parents=True, exist_ok=True)

    vp3dfile = output_dir / "vp3dfile.p"
    vs3dfile = output_dir / "vs3dfile.s"
    rho3dfile = output_dir / "rho3dfile.d"
    in_basin_mask_file = output_dir / "in_basin_mask.b"

    mode = "wb" if lat_ind == 0 else "ab"

    try:
        # If this is the first lat index, remove any existing files
        if lat_ind == 0:
            vp3dfile.unlink(missing_ok=True)
            vs3dfile.unlink(missing_ok=True)
            rho3dfile.unlink(missing_ok=True)
            in_basin_mask_file.unlink(missing_ok=True)
            logger.log(logging.DEBUG, f"Creating new VM files for emod3d: {output_dir}")

        # Flatten the arrays along the x and z dimensions. Write along x-axis first
        vp = vp_data.T.flatten()
        vs = vs_data.T.flatten()
        rho = rho_data.T.flatten()
        inbasin = inbasin_data.T.flatten()

        # Apply the minimum vs constraint
        vs = np.maximum(vs, min_vs)

        # Pack the data using the appropriate endianness
        vp_data = struct.pack(f"{endian_format}{len(vp)}f", *vp)
        vs_data = struct.pack(f"{endian_format}{len(vs)}f", *vs)
        rho_data = struct.pack(f"{endian_format}{len(rho)}f", *rho)
        inbasin_data = struct.pack(
            f"{endian_format}{len(inbasin)}f", *inbasin.astype(vp.dtype)
        )

        logger.log(
            logging.DEBUG,
            f"Writing global qualities to file for latitude index {lat_ind}",
        )

        # Write the binary data to files
        with open(vp3dfile, mode) as fvp:
            fvp.write(vp_data)

        with open(vs3dfile, mode) as fvs:
            fvs.write(vs_data)

        with open(rho3dfile, mode) as frho:
            frho.write(rho_data)

        with open(in_basin_mask_file, mode) as fmask:
            fmask.write(inbasin_data)

    except Exception as e:
        logger.log(logging.ERROR, f"Error writing emod3d data: {str(e)}")
        raise


def read_emomd3d_vm(output_dir: Path):
    """
    Read the write files into NumPy arrays.

    Parameters
    ----------
    output_dir : Path
        Directory containing the write files.

    Returns
    -------
    dict
        Dictionary containing the data from the write files.
    """
    files = {
        "vp": output_dir / "vp3dfile.p",
        "vs": output_dir / "vs3dfile.s",
        "rho": output_dir / "rho3dfile.d",
        "inbasin": output_dir / "in_basin_mask.b",
    }

    endianness = sys.byteorder
    endian_format = "<" if endianness == "little" else ">"

    data = {}
    for key, file in files.items():
        with open(file, "rb") as f:
            file_content = f.read()

            num_elements = len(file_content) // 4
            raw_data = np.array(
                struct.unpack(f"{endian_format}{num_elements}f", file_content),
                dtype=np.float32,
            )
            completed_mask = ~np.isnan(raw_data)
            if key == "inbasin":
                data[key] = raw_data.astype(np.int8)
            else:
                data[key] = raw_data[completed_mask]

    return data
