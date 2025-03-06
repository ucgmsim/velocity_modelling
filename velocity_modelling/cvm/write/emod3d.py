import struct
import sys

import numpy as np
from pathlib import Path

from velocity_modelling.cvm.geometry import PartialGlobalMesh
from velocity_modelling.cvm.velocity3d import PartialGlobalQualities
from velocity_modelling.cvm.logging import VMLogger


def write_global_qualities(
    output_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities: PartialGlobalQualities,
    vm_params: dict,
    lat_ind: int,
    logger: VMLogger = None,
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
    vm_params : dict
        Dictionary containing velocity model parameters.
    lat_ind : int
        Latitude index to determine the write mode (write or append).
    logger : VMLogger, optional
        Logger instance for logging messages.

    """
    if logger is None:
        logger = VMLogger("emod3d.wrote_global_qualities")

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
            logger.log(f"Creating new VM files for emod3d: {output_dir}", logger.INFO)

        # Flatten the arrays along the x and z dimensions. Write along x-axis first
        vp = partial_global_qualities.vp.T.flatten()
        vs = partial_global_qualities.vs.T.flatten()
        rho = partial_global_qualities.rho.T.flatten()
        inbasin = partial_global_qualities.inbasin.T.flatten()

        # Apply the minimum vs constraint
        vs = np.maximum(vs, vm_params["min_vs"])

        # Pack the data using the appropriate endianness
        vp_data = struct.pack(f"{endian_format}{len(vp)}f", *vp)
        vs_data = struct.pack(f"{endian_format}{len(vs)}f", *vs)
        rho_data = struct.pack(f"{endian_format}{len(rho)}f", *rho)
        inbasin_data = struct.pack(
            f"{endian_format}{len(inbasin)}f", *inbasin.astype(vp.dtype)
        )

        logger.log(
            f"Writing global qualities to file for latitude index {lat_ind}",
            logger.DEBUG,
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
        logger.log(f"Error writing emod3d data: {str(e)}", logger.ERROR)
        raise


def read_output_files(output_dir: Path):
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
