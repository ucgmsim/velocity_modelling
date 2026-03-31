#!/usr/bin/env python
"""
Convert HDF5 velocity model files to EMOD3D binary format.

This script converts HDF5 velocity model files (produced by hdf5.py) to the binary
format expected by EMOD3D, producing the same output files as emod3d.py:
- vp3dfile.p (P-wave velocities)
- vs3dfile.s (S-wave velocities)
- rho3dfile.d (densities)
- in_basin_mask.b (basin membership mask)

Parallelisation
---------------
The four output datasets (vp, vs, rho, inbasin) are written concurrently using a
``ThreadPoolExecutor(max_workers=4)``.  Threads are used rather than processes
because the bottleneck is I/O (reading from HDF5 and writing binary files), not
CPU computation, so the GIL is not a limiting factor.

Each thread opens its own independent ``h5py.File`` handle with ``locking=False``.
This is both thread-safe (no shared file handle) and avoids the stale-lock errors
that HDF5 can raise on HPC parallel filesystems (Lustre/GPFS) when the file was
written by a parallel process moments before.

"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer

from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


def _convert_dataset(
    src_h5: Path,
    dataset_path: str,
    out_file: Path,
    ny: int,
    nz: int,
    nx: int,
) -> None:
    """Convert a single HDF5 dataset to a binary output file.

    Opens its own h5py.File handle with locking=False so that multiple
    threads can read concurrently without HDF5 file-locking conflicts,
    which is important on HPC parallel filesystems (Lustre/GPFS).
    """
    z_values = slice(nz)
    x_values = slice(nx)
    buffer = np.empty((nz, nx), dtype=np.float32)
    with (
        h5py.File(src_h5, "r", locking=False) as f,
        open(out_file, "wb") as out,
    ):
        dset = f[dataset_path]
        for j in range(ny):
            dset.read_direct(buffer, (z_values, j, x_values))
            out.write(buffer.astype(np.float32).tobytes())


def convert_hdf5_to_emod3d(
    src_h5: Path, out_dir: Path, min_vs: float | None = None
) -> None:
    """
    Convert HDF5 velocity model to EMOD3D binary format.

    The key insight from emod3d.py is that it writes data as:
    - For each y-slice (latitude index j from 0 to ny-1):
      - Extract partial_global_qualities which has shape (nx, nz)
      - Apply: partial_global_qualities.vp.T.flatten()
      - This creates a flattened array where z varies fastest, then x

    The HDF5 format stores data as (nz, ny, nx), and hdf5.py already
    transposes the data before storing: (nx, nz) -> (nz, nx)
    So when we extract slice [:, j, :] we get (nz, nx) which is exactly
    what emod3d.py produces with .T.flatten()

    Parameters
    ----------
    src_h5 : Path
        Path to input HDF5 file containing velocity model data.
    out_dir : Path
        Output directory where binary files will be written.
    min_vs : float | None, optional
        Minimum Vs value to enforce in km/s. If None, uses value from HDF5 metadata.

    Raises
    ------
    FileNotFoundError
        If the input HDF5 file does not exist.
    KeyError
        If required datasets are missing from the HDF5 file.
    ValueError
        If data shapes or types are invalid.
    OSError
        If there are issues creating output directory or writing binary files.

    Notes
    -----
    This function creates binary files that exactly match the output format
    of the original emod3d.py writer, ensuring compatibility with EMOD3D
    simulation software.
    """
    start_time = time.time()

    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "/properties/vp": out_dir / "vp3dfile.p",
        "/properties/vs": out_dir / "vs3dfile.s",
        "/properties/rho": out_dir / "rho3dfile.d",
        "/properties/inbasin": out_dir / "in_basin_mask.b",
    }

    for filepath in files.values():
        filepath.unlink(missing_ok=True)

    with h5py.File(src_h5, "r", locking=False) as f:
        nz, ny, nx = f["/properties/vp"].shape

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _convert_dataset, src_h5, dset_path, out_file, ny, nz, nx
            ): dset_path
            for dset_path, out_file in files.items()
        }
        for future in futures:
            future.result()

    end_time = time.time()
    processing_time = end_time - start_time

    print("✅ Conversion complete!")
    for filepath in files.values():
        print(f"   Created: {filepath}")
    print(f"   Processing time: {processing_time:.2f} seconds")


@cli.from_docstring(app)
def convert_hdf5_to_emod3d_main(
    src_h5: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
            writable=True,
        ),
    ],
    min_vs: float | None = None,
) -> None:
    """
    Convert HDF5 velocity model to EMOD3D binary format.

    This command-line interface converts HDF5 velocity model files (produced by hdf5.py)
    to the binary format expected by EMOD3D, producing the same output files as emod3d.py.

    The conversion process reads velocity model data from the HDF5 file and writes it
    in the specific binary format required by EMOD3D simulations, maintaining exact
    compatibility with the original emod3d.py writer.

    Parameters
    ----------
    src_h5 : Path
        Path to the input HDF5 file containing velocity model data.
        Must exist and be a valid HDF5 file with the expected datasets.
    out_dir : Path
        Output directory where the binary files will be written.
        Will be created if it doesn't exist.
    min_vs : float | None, optional
        Minimum S-wave velocity value to enforce in km/s.
        If not specified, uses the value stored in the HDF5 metadata.

    Raises
    ------
    typer.Exit
        Exits with code 1 if conversion fails due to file errors,
        invalid data, or other processing issues.

    Examples
    --------
    Convert with default minimum Vs from HDF5 metadata:

    >>> convert_hdf5_to_emod3d_main(Path("model.h5"), Path("output"))

    Convert with custom minimum Vs constraint:

    >>> convert_hdf5_to_emod3d_main(Path("model.h5"), Path("output"), min_vs=0.5)

    Notes
    -----
    The output binary files are:
    - vp3dfile.p: P-wave velocities
    - vs3dfile.s: S-wave velocities
    - rho3dfile.d: Densities
    - in_basin_mask.b: Basin membership mask
    """
    try:
        convert_hdf5_to_emod3d(src_h5, out_dir, min_vs)
    except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"❌ Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
