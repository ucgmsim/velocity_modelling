import argparse
import struct
import sys
from logging import Logger
import logging

import numpy as np
import yaml
from pathlib import Path

from velocity_modelling.cvm.geometry import PartialGlobalMesh
from velocity_modelling.cvm.velocity3d import PartialGlobalQualities


def write_global_qualities(
    output_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities: PartialGlobalQualities,
    vm_params: dict,
    lat_ind: int,
    logger: Logger,
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
    logger : Logger
        Logger instance for logging messages.

    Returns
    -------
    None
    """

    # perform endian check
    endianness = sys.byteorder

    endian_format = "<" if endianness == "little" else ">"

    output_dir.mkdir(parents=True, exist_ok=True)

    vp3dfile = output_dir / "vp3dfile.p"
    vs3dfile = output_dir / "vs3dfile.s"
    rho3dfile = output_dir / "rho3dfile.d"
    in_basin_mask_file = output_dir / "in_basin_mask.b"

    mode = "wb" if lat_ind == 0 else "ab"

    # If this is the first lat index, remove any existing files
    if lat_ind == 0:
        vp3dfile.unlink(missing_ok=True)
        vs3dfile.unlink(missing_ok=True)
        rho3dfile.unlink(missing_ok=True)
        in_basin_mask_file.unlink(missing_ok=True)

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

    # Write the binary data to files
    with open(vp3dfile, mode) as fvp:
        fvp.write(vp_data)

    with open(vs3dfile, mode) as fvs:
        fvs.write(vs_data)

    with open(rho3dfile, mode) as frho:
        frho.write(rho_data)

    with open(in_basin_mask_file, mode) as fmask:
        fmask.write(inbasin_data)


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


def compare_output_files(
    dir1: Path, dir2: Path, nx: int, ny: int, nz: int, threshold=1e-5
):
    data1 = read_output_files(dir1)
    data2 = read_output_files(dir2)

    comparison = {}
    for key in data1:
        if key in data2:
            min_length = min(len(data1[key]), len(data2[key]))
            print(
                f"Comparing {key} with length {min_length} Data1 len={len(data1[key])} Data2 len={len(data2[key])}"
            )
            data1_trimmed = data1[key][:min_length]
            data2_trimmed = data2[key][:min_length]
            print(
                f"Data1: max={np.max(data1_trimmed)}, min={np.min(data1_trimmed)} mean={np.mean(data1_trimmed)} std={np.std(data1_trimmed)}"
            )
            print(
                f"Data2: max={np.max(data2_trimmed)}, min={np.min(data2_trimmed)} mean={np.mean(data2_trimmed)} std={np.std(data2_trimmed)}"
            )

            difference = data1_trimmed - data2_trimmed

            significant_diff_indices = np.where(np.abs(difference) > threshold)[0]

            if significant_diff_indices.size > 0:
                first_significant_index = significant_diff_indices[0]
                x_index = first_significant_index % nx
                z_index = (first_significant_index // nx) % nz
                y_index = first_significant_index // (nx * nz)

            else:
                x_index = y_index = z_index = first_significant_index = None

            comparison[key] = {
                "allclose": np.allclose(data1_trimmed, data2_trimmed),
                "difference": np.abs(difference),
                "max_diff": np.max(np.abs(difference)),
                "min_diff": np.min(np.abs(difference)),
                "mean_diff": np.mean(np.abs(difference)),
                "std_diff": np.std(np.abs(difference)),
                "first_significant_index": first_significant_index,
                "x_index": x_index,
                "y_index": y_index,
                "z_index": z_index,
            }

        else:
            comparison[key] = "File missing in second directory"

    return comparison


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare write files from two directories."
    )
    parser.add_argument(
        "output_dir1", type=Path, help="First directory containing the write files."
    )
    parser.add_argument(
        "output_dir2", type=Path, help="Second directory containing the write files."
    )
    parser.add_argument("vm_params", type=Path, help="Path to the vm_params.yaml file")
    args = parser.parse_args()

    output_dir1 = args.output_dir1
    output_dir2 = args.output_dir2
    vm_params_path = args.vm_params

    with open(vm_params_path, "r") as f:
        vm_params = yaml.safe_load(f)

    nx = vm_params["nx"]
    ny = vm_params["ny"]
    nz = vm_params["nz"]

    comparison_results = compare_output_files(output_dir1, output_dir2, nx, ny, nz)
    for key in comparison_results:
        print(f"Results for {key}:")
        print(comparison_results[key])
