import struct
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
    endian_int = struct.unpack("<I", struct.pack("=I", 1))[0] == 1
    endian_format = "<" if endian_int else ">"

    vp3dfile = output_dir / "vp3dfile.p"
    vs3dfile = output_dir / "vs3dfile.s"
    rho3dfile = output_dir / "rho3dfile.d"
    in_basin_mask_file = output_dir / "in_basin_mask.b"

    mode = "wb" if lat_ind == 0 else "ab"

    # bsize = partial_global_mesh.nx * partial_global_mesh.nz
    # vp = np.zeros(bsize, dtype=np.float32)
    # vs = np.zeros(bsize, dtype=np.float32)
    # rho = np.zeros(bsize, dtype=np.float32)
    # inbasin = np.zeros(bsize, dtype=np.float32)

    with (
        open(vp3dfile, mode) as fvp,
        open(vs3dfile, mode) as fvs,
        open(rho3dfile, mode) as frho,
        open(in_basin_mask_file, mode) as fmask,
    ):

        for iz in range(partial_global_mesh.nz):
            for ix in range(partial_global_mesh.nx):
                vs_temp = np.max(
                    [partial_global_qualities.vs[ix][iz], vm_params["min_vs"]]
                )
                vp_temp = partial_global_qualities.vp[ix][iz]
                rho_temp = partial_global_qualities.rho[ix][iz]
                inbasin_temp = partial_global_qualities.inbasin[ix][iz]

                vs_write = struct.pack(f"{endian_format}f", vs_temp)
                vp_write = struct.pack(f"{endian_format}f", vp_temp)
                rho_write = struct.pack(f"{endian_format}f", rho_temp)
                inbasin_write = struct.pack(f"{endian_format}f", inbasin_temp)

                fvp.write(vp_write)
                fvs.write(vs_write)
                frho.write(rho_write)
                fmask.write(inbasin_write)


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
    endian_int = struct.unpack("<I", struct.pack("=I", 1))[0] == 1
    endian_format = "<" if endian_int else ">"

    data = {}
    for key, file in files.items():
        with open(file, "rb") as f:
            file_content = f.read()
            num_elements = len(file_content) // 4

            data[key] = np.array(
                struct.unpack(f"{endian_format}{num_elements}f", file_content),
                dtype=np.float32,
            )

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
        difference = data1[key] - data2[key]
        comparison[key] = {
            "allclose": np.allclose(data1[key], data2[key]),
            "difference": np.abs(difference),
            "max_difference": np.max(np.abs(difference)),
            "average_difference": np.mean(np.abs(difference)),
            "std_difference": np.std(difference),
        }

    return comparison


if __name__ == "__main__":
    output_dir1 = Path(
        "/home/seb56/velocity_modelling/velocity_modelling/benchmark/RangipoS/tmp"
    )  # Python
    output_dir2 = Path(
        "/home/seb56/velocity_modelling/velocity_modelling/benchmark/RangipoS/tmp/output/Velocity_Model"
    )  # C
    comparison_results = compare_output_files(output_dir1, output_dir2)
    print(comparison_results)
