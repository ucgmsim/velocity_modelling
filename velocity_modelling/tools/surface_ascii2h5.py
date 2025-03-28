"""
Convert ASCII format surface file (.in) from plain text format to HDF5.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def ascii_to_hdf5(input_file_path: str | Path, output_file_path: str | Path):
    """
    Convert Digital Elevation Model (DEM) from plain text format to HDF5.

    Parameters
    ----------
    input_file_path : str or Path
        Path to the input DEM file.
    output_file_path : str or Path
        Path to the output HDF5 file.
    """
    # Ensure paths are Path objects
    input_file_path = Path(input_file_path)
    output_file_path = Path(output_file_path)

    print(f"Converting {input_file_path} to {output_file_path}")

    try:
        # Read the dimensions, latitude and longitude values
        with open(input_file_path, "r") as f:
            # Read dimensions
            dimensions = f.readline().strip().split()
            nrows, ncols = int(dimensions[0]), int(dimensions[1])

            # Read latitude values (full array)
            lat_line = f.readline().strip().split()
            lat_values = np.array([float(x) for x in lat_line])
            
            # Read longitude values (full array)
            lon_line = f.readline().strip().split()
            lon_values = np.array([float(x) for x in lon_line])

            # Read elevation data - one row at a time
            elevation_data = np.zeros((nrows, ncols))
            for i in range(nrows):
                # Read lines until we have enough values for this row
                row_data = []
                while len(row_data) < ncols:
                    line = f.readline().strip().split()
                    if not line:  # Handle end of file
                        break
                    row_data.extend(line)
                
                # Fill the row with as many values as we have (up to ncols)
                for j in range(min(ncols, len(row_data))):
                    elevation_data[i, j] = float(row_data[j])

        # Create the output directory if it doesn't exist
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create HDF5 file
        with h5py.File(output_file_path, "w") as hf:
            # Store metadata
            hf.attrs["nrows"] = nrows
            hf.attrs["ncols"] = ncols

            # Create datasets with compression
            hf.create_dataset("latitude", data=lat_values, compression="gzip")
            hf.create_dataset("longitude", data=lon_values, compression="gzip")
            hf.create_dataset("elevation", data=elevation_data, compression="gzip")

        print(f"Conversion complete. HDF5 file saved to {output_file_path}")

    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            raise  # Re-raise critical exceptions
        raise ValueError(f"Error during conversion: {str(e)}")


def main():
    """
    Main function for command-line interface.

    """
    parser = argparse.ArgumentParser(description="Convert DEM file to HDF5 format")
    parser.add_argument("input_file", type=str, help="Path to the input DEM file")
    parser.add_argument(
        "--out-dir", type=str, help="Output directory (default is same as input file)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{input_path.stem}.h5"
    else:
        output_path = input_path.with_suffix(".h5")

    ascii_to_hdf5(input_path, output_path)


if __name__ == "__main__":
    main()
