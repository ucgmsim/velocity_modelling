#!/usr/bin/env python3
"""
Convert ASCII tomography files to HDF5 format (with dtype/compression options).

This script converts ASCII tomography files to HDF5 format. The input directory contains
files with names like "surf_tomography_vp_elev0p25.in", "surf_tomography_vs_elev0p25.in" etc, where
"vp", "vs", and "rho" are the velocity types and "0p25" is the elevation. The script reads the
elevation values from the file names and ensures that they match across all velocity types. The
output HDF5 file will contain groups for each elevation, with datasets for latitudes, longitudes,
and the velocity types.

Example usage:
python tomo_ascii2h5.py /path/to/tomography_files output_name --out-dir /path/to/output_dir --dtype f32 --gzip 6 --chunk 128x128

This will convert the tomography files in /path/to/tomography_files to a file named output_name.h5
in the same directory. If --out-dir is specified, the output file will be saved in that directory
instead.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer

from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


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
    elevations_by_type: dict[str, set[float]] = {v: set() for v in vtypes}
    for filename in input_dir.glob("surf_tomography_*.in"):
        match = elev_pattern.match(filename.name)
        if match:
            vtype, elev_str = match.groups()
            elev = float(elev_str.replace("p", "."))
            elevations_by_type[vtype].add(elev)
            print(f"Found file: {filename.name}, vtype: {vtype}, elev: {elev}")
    if (
        not elevations_by_type["vp"]
        or not elevations_by_type["vs"]
        or not elevations_by_type["rho"]
    ):
        missing = [v for v, s in elevations_by_type.items() if not s]
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
    return elevations_by_type["vp"]


def chunk_from_shape(
    shape: tuple[int, ...], chunk_2d: tuple[int, int]
) -> tuple[int, ...] | bool:
    """
    Determine chunk size for HDF5 datasets based on shape and 2D chunk size.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the dataset (1D or 2D).

    chunk_2d : tuple[int, int]
        Desired chunk size for 2D datasets (H, W).

    Returns
    -------
    tuple[int, ...] | bool
        Chunk size as a tuple, or True if no chunking is needed.

    """

    if len(shape) == 2:
        return (min(chunk_2d[0], shape[0]), min(chunk_2d[1], shape[1]))
    if len(shape) == 1:
        # 1D coords: let h5py decide or use a modest chunk
        n = shape[0]
        return (min(4096, max(256, n // 8)),)
    return True


def convert_ascii_to_hdf5(
    input_dir: Path,
    name: str,
    out_dir: Path | None = None,
    dtype_opt: str = "f64",  # "f64" or "f32" for vp/vs/rho
    gzip_level: int = 4,  # 1..9
    shuffle: bool = True,
    chunk_2d: tuple[int, int] = (256, 256),
):
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
    dtype_opt : str, optional
        Data dtype for vp/vs/rho: "f64" (default) or "f32".
    gzip_level : int, optional
        Gzip compression level 1..9 (default 4).
    shuffle : bool, optional
        Enable shuffle filter (default is True).
    chunk_2d : tuple[int, int], optional
        2D chunk size as (H, W) tuple (default (256, 256)). This controls how data is chunked in the HDF5 file.

    """

    vtypes = ["vp", "vs", "rho"]
    elevations = sorted(get_elevations_from_files(input_dir))
    if not elevations:
        raise ValueError(f"No valid tomography files found in {input_dir}")
    input_path = Path(input_dir)

    # Output path
    if out_dir is None:
        output_path = input_path / f"{name}.h5"
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(out_dir) / f"{name}.h5"

    # Dtype policy
    data_dtype = np.float64 if dtype_opt.lower() == "f64" else np.float32
    coord_dtype = np.float64  # keep coords precise

    with h5py.File(output_path, "w") as h5f:
        # file-level metadata to advertise decisions
        h5f.attrs["created"] = datetime.utcnow().isoformat() + "Z"
        h5f.attrs["generator"] = "tomo_ascii2h5.py"
        h5f.attrs["schema"] = "NZTomographyLevelStacked v1"
        h5f.attrs["data_dtype_vp_vs_rho"] = (
            "float32" if data_dtype == np.float32 else "float64"
        )
        h5f.attrs["coord_dtype_lat_lon"] = "float64"
        h5f.attrs["compression"] = f"gzip:{gzip_level}"
        h5f.attrs["shuffle"] = bool(shuffle)
        h5f.attrs["chunk_2d"] = chunk_2d

        for elev in elevations:
            elev_file_str = (
                str(int(elev)) if elev == int(elev) else f"{elev:.2f}".replace(".", "p")
            )
            elev_group_name = str(int(elev)) if elev == int(elev) else f"{elev:.2f}"
            print(
                f"Processing elevation {elev_group_name} (files suffix: {elev_file_str})"
            )
            g = h5f.create_group(elev_group_name)

            # Load coords from rho file
            ref_file = input_path / f"surf_tomography_rho_elev{elev_file_str}.in"
            if not ref_file.exists():
                print(f"Warning: missing {ref_file}, skipping elevation {elev}")
                continue
            with open(ref_file, "r") as f:
                nlat, nlon = map(int, f.readline().split())
                latitudes = np.array(
                    [float(x) for x in f.readline().split()], dtype=coord_dtype
                )
                longitudes = np.array(
                    [float(x) for x in f.readline().split()], dtype=coord_dtype
                )

            # coords
            g.create_dataset(
                "latitudes",
                data=latitudes,
                compression="gzip",
                compression_opts=gzip_level,
                shuffle=shuffle,
                chunks=chunk_from_shape(latitudes.shape, chunk_2d),
            )
            g.create_dataset(
                "longitudes",
                data=longitudes,
                compression="gzip",
                compression_opts=gzip_level,
                shuffle=shuffle,
                chunks=chunk_from_shape(longitudes.shape, chunk_2d),
            )

            # fields
            for vtype in vtypes:
                filename = (
                    input_path / f"surf_tomography_{vtype}_elev{elev_file_str}.in"
                )
                if not filename.exists():
                    print(f"Warning: missing {filename}")
                    continue
                with open(filename, "r") as f:
                    f.readline()
                    f.readline()
                    f.readline()
                    arr = np.array(
                        [float(x) for x in f.read().split()], dtype=data_dtype
                    )
                    arr = arr.reshape(nlat, nlon)

                dset = g.create_dataset(
                    vtype,
                    data=arr,
                    compression="gzip",
                    compression_opts=gzip_level,
                    shuffle=shuffle,
                    chunks=chunk_from_shape(arr.shape, chunk_2d),
                )
                dset.attrs["units"] = (
                    "km/s" if vtype in ("vp", "vs") else "g/cc"
                )  # adjust if needed
                dset.attrs["dtype"] = (
                    "float32" if data_dtype == np.float32 else "float64"
                )

                print(f"Wrote {vtype} at elevation {elev_group_name} to {output_path}")
    print(f"Done: {output_path}")


@cli.from_docstring(app)
def convert_tomo_to_h5(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Input directory containing ASCII tomography files",
        ),
    ],
    name: Annotated[
        str,
        typer.Argument(help="Base name for the output HDF5 file (without extension)"),
    ],
    out_dir: Annotated[
        Path | None,
        typer.Option(file_okay=False, help="Output directory for the HDF5 file"),
    ] = None,
    dtype: Annotated[
        str, typer.Option(help="Data dtype for vp/vs/rho: f32 (default) or f64")
    ] = "f32",
    gzip: Annotated[int, typer.Option(help="gzip level 1..9 (default 4)")] = 4,
    no_shuffle: Annotated[bool, typer.Option(help="Disable shuffle filter")] = False,
    chunk: Annotated[
        str, typer.Option(help="2D chunk as HxW, default 256x256")
    ] = "256x256",
):
    """
    Convert ASCII tomography files to HDF5 format.

    This tool consolidates multiple ASCII tomography files into a single,
    structured HDF5 file. It handles files with various elevation values
    and velocity parameters (vp, vs, rho) and ensures they are consistent.

    Parameters
    ----------
    input_dir : Path
        Path to directory containing ASCII tomography files.
    name : str
        Name for the output HDF5 file (without extension).
    out_dir : Path, optional
        Output directory for the HDF5 file (if unspecified, saves to input_dir/name.h5).
    dtype : str, optional
        Data dtype for vp/vs/rho: "f64" (default) or "f32".
    gzip : int, optional
        Gzip compression level 1..9 (default 4).
    no_shuffle : bool, optional
        Disable shuffle filter (default is enabled).
    chunk : str, optional
        2D chunk size as "HxW" (default "256x256"). This controls how data is chunked in the HDF5 file.

    """

    h, w = (int(s) for s in chunk.lower().split("x"))
    convert_ascii_to_hdf5(
        input_dir,
        name,
        out_dir,
        dtype_opt=dtype,
        gzip_level=gzip,
        shuffle=(not no_shuffle),
        chunk_2d=(h, w),
    )


if __name__ == "__main__":
    app()
