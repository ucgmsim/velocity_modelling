"""
Convert ASCII tomography files to HDF5 format.

This script converts ASCII tomography files to HDF5 format. The input directory contains
files with names like "surf_tomography_vp_elev0p25.in", "surf_tomography_vs_elev0p25.in" etc, where
"vp", "vs", and "rho" are the velocity types and "0p25" is the elevation. The script reads the
elevation values from the file names and ensures that they match across all velocity types. The
output HDF5 file will contain groups for each elevation, with datasets for latitudes, longitudes,
and the velocity types.

Example usage:
    python tomo_ascii2h5.py /path/to/tomography_files output_name --out-dir /path/to/output_dir

This will convert the tomography files in /path/to/tomography_files to a file named output_name.h5
in the same directory. If --out-dir is specified, the output file will be saved in that directory
instead.


"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np


def get_elevations_from_files(input_dir: Path) -> set[float]:
    """
    Extract unique elevation values from file names in input_dir.

    Parameters
    ----------
    input_dir : Path
        Path to directory containing tomography files.

    Returns
    -------
    set[float]
        Set of unique elevation values found in file names.

    Raises
    ------
    ValueError
        If no elevation files are found for any velocity type, or if elevations do not match
        across all velocity types.

    """
    vtypes = ["vp", "vs", "rho"]
    elev_pattern = re.compile(r"surf_tomography_(vp|vs|rho)_elev(-?\d+(?:p\d+)?)\.in")

    elevations_by_type: dict[str, set[float]] = {vtype: set() for vtype in vtypes}

    for filename in input_dir.glob("surf_tomography_*.in"):
        match = elev_pattern.match(filename.name)
        if match:
            vtype, elev_str = match.groups()
            # Convert elevation string (e.g., "0p25" or "-750") to float
            elev = float(elev_str.replace("p", "."))
            elevations_by_type[vtype].add(elev)
            print(f"Found file: {filename.name}, vtype: {vtype}, elev: {elev}")

    # Check if elevations match across all types
    if (
        not elevations_by_type["vp"]
        or not elevations_by_type["vs"]
        or not elevations_by_type["rho"]
    ):
        missing = [vtype for vtype, elevs in elevations_by_type.items() if not elevs]
        raise ValueError(
            f"No elevation files found for {', '.join(missing)} in {input_dir}"
        )

    if (
        elevations_by_type["vp"] != elevations_by_type["vs"]
        or elevations_by_type["vp"] != elevations_by_type["rho"]
    ):
        raise ValueError(
            "Elevations do not match across vp, vs, and rho:\n"
            f"vp: {sorted(elevations_by_type['vp'])}\n"
            f"vs: {sorted(elevations_by_type['vs'])}\n"
            f"rho: {sorted(elevations_by_type['rho'])}"
        )

    return elevations_by_type["vp"]  # All match, so any type's set is fine


def convert_ascii_to_hdf5(input_dir: Path, name: str, out_dir: Path = None):
    """
    Convert ASCII tomography files to HDF5 format.

    Parameters
    ----------
    input_dir : Path
        Path to directory containing ASCII tomography files
    name : str
        Name for the output HDF5 file (without extension)

    out_dir : Path, optional
        Output directory for the HDF5 file (if unspecified, saves to input_dir/name.h5)

    """

    vtypes = ["vp", "vs", "rho"]

    # Get elevations from files and ensure they match
    elevations = sorted(get_elevations_from_files(input_dir))
    if not elevations:
        raise ValueError(f"No valid tomography files found in {input_dir}")

    print(f"Found matching elevations: {elevations}")

    input_path = Path(input_dir)

    # Determine output file path
    if out_dir is None:
        output_path = input_path / f"{name}.h5"
    else:
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"{name}.h5"

    with h5py.File(output_path, "w") as h5f:
        for elev in elevations:
            # Convert elevation to string for file naming (e.g., "-750" or "0p25")
            if elev == int(elev):
                elev_file_str = str(int(elev))  # e.g., "-750"
            else:
                elev_file_str = f"{elev:.2f}".replace(".", "p")  # e.g., "0p25"

            # Use consistent format for HDF5 group name (e.g., "-750" or "0.25")
            elev_group_name = str(int(elev)) if elev == int(elev) else f"{elev:.2f}"
            elev_group = h5f.create_group(elev_group_name)

            # Load coordinates from one file (using rho as reference)
            ref_file = input_path / f"surf_tomography_rho_elev{elev_file_str}.in"
            print(f"Checking for reference file: {ref_file}")
            if not ref_file.exists():
                print(
                    f"Warning: Reference file {ref_file} not found, skipping elevation {elev}"
                )
                continue

            with open(ref_file, "r") as f:
                nlat, nlon = map(int, f.readline().split())
                latitudes = np.array([float(x) for x in f.readline().split()])
                longitudes = np.array([float(x) for x in f.readline().split()])

            elev_group.create_dataset("latitudes", data=latitudes)
            elev_group.create_dataset("longitudes", data=longitudes)

            # Store each velocity type
            for vtype in vtypes:
                filename = (
                    input_path / f"surf_tomography_{vtype}_elev{elev_file_str}.in"
                )
                print(f"Checking for file: {filename}")
                if not filename.exists():
                    print(f"Warning: File {filename} not found")
                    continue

                with open(filename, "r") as f:
                    f.readline()  # Skip header
                    f.readline()  # Skip latitudes
                    f.readline()  # Skip longitudes
                    data = np.array([float(x) for x in f.read().split()])
                    data = data.reshape(nlat, nlon)

                elev_group.create_dataset(vtype, data=data)
                print(f"Converted {vtype} at elevation {elev} to {output_path}")


def main():
    """
    Convert ASCII tomography files to HDF5 format.

    """
    parser = argparse.ArgumentParser(
        description="Convert ASCII tomography files to HDF5 format."
    )
    parser.add_argument(
        "input_dir", type=Path, help="Input directory containing ASCII tomography files"
    )
    parser.add_argument(
        "name", type=str, help="Name for the output HDF5 file (without extension)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for the HDF5 file (if unspecified, saves to input_dir/name.h5)",
    )

    args = parser.parse_args()
    convert_ascii_to_hdf5(args.input_dir, args.name, args.out_dir)


if __name__ == "__main__":
    main()
