"""
Convert binary velocity model files to CSV format.

This is still experimental

TODO:  emod3d output doesn't have lat/lon/depth information, and this script only works out (y,x,z) coordinates
We need an auxiliary output from generate_velocity_model.py to map the (y,x,z) coordinates to lat/lon/depth

"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import struct
import argparse


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


def write_to_csv(output_dir: Path, ny: int, nx: int, nz: int, csv_file: Path):
    """
    Read binary velocity model files and write to a CSV file with y, x, z, Vp, Vs, Rho.

    Parameters
    ----------
    output_dir : Path
        Directory containing the binary files.
    ny : int
        Number of points along the y-axis.
    nx : int
        Number of points along the x-axis.
    nz : int
        Number of points along the z-axis.
    csv_file : Path
        Path to the output CSV file.
    """
    # Read the binary files
    data = read_output_files(output_dir)

    # Check that the total number of elements matches ny * nx * nz
    expected_elements = ny * nx * nz
    for key, array in data.items():
        if len(array) != expected_elements:
            raise ValueError(
                f"Data length mismatch for {key}: expected {expected_elements}, got {len(array)}"
            )

    # Reshape the flattened arrays back to (ny, nx, nz)
    vp = data["vp"].reshape((ny, nx, nz))
    vs = data["vs"].reshape((ny, nx, nz))
    rho = data["rho"].reshape((ny, nx, nz))

    # Generate coordinate grids
    y_indices, x_indices, z_indices = np.indices((ny, nx, nz))

    # Flatten the arrays for CSV output
    y_flat = y_indices.flatten()
    x_flat = x_indices.flatten()
    z_flat = z_indices.flatten()
    vp_flat = vp.flatten()
    vs_flat = vs.flatten()
    rho_flat = rho.flatten()

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "y": y_flat,
            "x": x_flat,
            "z": z_flat,
            "Vp": vp_flat,
            "Vs": vs_flat,
            "Rho": rho_flat,
        }
    )

    # Ensure the output directory exists
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    # Write to CSV
    df.to_csv(csv_file, index=False)
    print(f"CSV file written to {csv_file}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert binary velocity model files to CSV format."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing the binary files (vp3dfile.p, vs3dfile.s, rho3dfile.d, in_basin_mask.b)",
    )
    parser.add_argument(
        "--nx", type=int, required=True, help="Number of points along the x-axis"
    )
    parser.add_argument(
        "--ny", type=int, required=True, help="Number of points along the y-axis"
    )
    parser.add_argument(
        "--nz", type=int, required=True, help="Number of points along the z-axis"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output CSV file (defaults to input_dir/velocity_model.csv)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Convert input directory to Path object
    output_dir = Path(args.input_dir)

    # Set default output file if not provided
    csv_file = Path(args.output) if args.output else output_dir / "velocity_model.csv"

    # Call the conversion function
    write_to_csv(output_dir, args.ny, args.nx, args.nz, csv_file)


if __name__ == "__main__":
    main()
