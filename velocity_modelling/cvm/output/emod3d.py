import argparse
import struct
import sys
import numpy as np

from pathlib import Path
from velocity_modelling.cvm.registry import (
    PartialGlobalMesh,
    PartialGlobalQualities,
    PartialGlobalQualities,
    Logger,
)


def write_global_qualities(
    output_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities,
    vm_params: dict,
    lat_ind: int,
    logger: Logger,
):
    """
    Purpose: write the full velocity model to file

    Input variables:
    partial_global_mesh        - pointer to structure containing lat lon grid
    partial_global_qualities   - pointer to structure containing vp vs and rho for all gridpoints

    Output variables:
    N/A.
    """

    # perform endian check
    endianness = sys.byteorder

    endian_format = "<" if endianness == "little" else ">"

    vp3dfile = output_dir / "vp3dfile.p"
    vs3dfile = output_dir / "vs3dfile.s"
    rho3dfile = output_dir / "rho3dfile.d"
    in_basin_mask_file = output_dir / "in_basin_mask.b"

    mode = "wb" if lat_ind == 0 else "ab"

    # Flatten the arrays along the x and z dimensions
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
    inbasin_data = struct.pack(f"{endian_format}{len(inbasin)}f", *inbasin)

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
    Read the output files into NumPy arrays.

    Parameters
    ----------
    output_dir : Path
        Directory containing the output files.

    Returns
    -------
    dict
        Dictionary containing the data from the output files.
    """
    files = {
        "vp": output_dir / "vp3dfile.p",
        "vs": output_dir / "vs3dfile.s",
        "rho": output_dir / "rho3dfile.d",
        "inbasin": output_dir / "in_basin_mask.b",
    }

    # Check endianness

    endianness = sys.byteorder
    endian_format = "<" if endianness == "little" else ">"

    data = {}
    for key, file in files.items():
        with open(file, "rb") as f:
            file_content = f.read()
            num_elements = len(file_content) // 4  # 4 bytes per float

            raw_data = np.array(
                struct.unpack(f"{endian_format}{num_elements}f", file_content),
                dtype=np.float32,
            )
            # Identify completed parts (assuming NaNs indicate incomplete parts)
            completed_mask = ~np.isnan(raw_data)
            data[key] = raw_data[completed_mask]

    return data


def compare_output_files(dir1: Path, dir2: Path):
    """
    Compare the output files from two directories.

    Parameters
    ----------
    dir1 : Path
        First directory containing the output files.
    dir2 : Path
        Second directory containing the output files.

    Returns
    -------
    dict
        Dictionary containing the comparison results.
    """
    data1 = read_output_files(dir1)
    data2 = read_output_files(dir2)

    comparison = {}
    for key in data1:
        if key in data2:
            min_length = min(len(data1[key]), len(data2[key]))
            print(f"Comparing {key} with length {min_length}")
            data1_trimmed = data1[key][:min_length]
            data2_trimmed = data2[key][:min_length]
            print(
                f"Data1: max={np.max(data1_trimmed)}, min={np.min(data1_trimmed)} mean={np.mean(data1_trimmed)} std={np.std(data1_trimmed)}"
            )
            print(
                f"Data2: max={np.max(data2_trimmed)}, min={np.min(data2_trimmed)} mean={np.mean(data2_trimmed)} std={np.std(data2_trimmed)}"
            )

            difference = data1_trimmed - data2_trimmed
            comparison[key] = {
                "allclose": np.allclose(data1_trimmed, data2_trimmed),
                "difference": np.abs(difference),
                "max_diff": np.max(np.abs(difference)),
                "min_diff": np.min(np.abs(difference)),
                "mean_diff": np.mean(np.abs(difference)),
                "std_diff": np.std(np.abs(difference)),
            }
        else:
            comparison[key] = "File missing in second directory"

    return comparison


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare output files from two directories."
    )
    parser.add_argument(
        "output_dir1", type=Path, help="First directory containing the output files."
    )
    parser.add_argument(
        "output_dir2", type=Path, help="Second directory containing the output files."
    )
    args = parser.parse_args()

    output_dir1 = args.output_dir1
    output_dir2 = args.output_dir2

    comparison_results = compare_output_files(output_dir1, output_dir2)
    for key in comparison_results:
        print(f"Results for {key}:")
        print(comparison_results[key])
