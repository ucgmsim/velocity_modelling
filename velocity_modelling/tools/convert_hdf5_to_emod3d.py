#!/usr/bin/env python
"""
Convert HDF5 velocity model files to EMOD3D binary format.

This script converts HDF5 velocity model files (produced by hdf5.py) to the binary
format expected by EMOD3D, producing the same output files as emod3d.py:
- vp3dfile.p (P-wave velocities)
- vs3dfile.s (S-wave velocities)
- rho3dfile.d (densities)
- in_basin_mask.b (basin membership mask)
"""

import struct
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import h5py
import numpy as np
import typer
from qcore import cli
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_enable=False)


def convert_hdf5_to_emod3d(
        src_h5: Path, out_dir: Path, min_vs: Optional[float] = None
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
        Path to input HDF5 file.
    out_dir : Path
        Output directory for binary files.
    min_vs : float, optional
        Minimum Vs value to enforce. If None, uses value from HDF5 metadata.
    """
    start_time = time.time()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine endianness and format string (match emod3d.py exactly)
    endianness = sys.byteorder
    endian_format = "<" if endianness == "little" else ">"

    # Output file paths (match emod3d.py exactly)
    vp3dfile = out_dir / "vp3dfile.p"
    vs3dfile = out_dir / "vs3dfile.s"
    rho3dfile = out_dir / "rho3dfile.d"
    in_basin_mask_file = out_dir / "in_basin_mask.b"

    with h5py.File(src_h5, "r") as f:
        # Read data arrays - HDF5 format is (nz, ny, nx)
        vp_full = np.array(f["/properties/vp"])  # (nz, ny, nx)
        vs_full = np.array(f["/properties/vs"])  # (nz, ny, nx)
        rho_full = np.array(f["/properties/rho"])  # (nz, ny, nx)
        inbasin_full = np.array(f["/properties/inbasin"])  # (nz, ny, nx)

        nz, ny, nx = vp_full.shape

        # Get min_vs from file metadata if not provided
        if min_vs is None:
            min_vs = f["/properties/vs"].attrs.get("min_value_enforced", 0.0)

        print(f"Converting HDF5 model with shape (nz={nz}, ny={ny}, nx={nx})")
        print(f"Applying minimum Vs constraint: {min_vs} km/s")

    # Remove any existing files (match emod3d.py behavior)
    for filepath in [vp3dfile, vs3dfile, rho3dfile, in_basin_mask_file]:
        filepath.unlink(missing_ok=True)

    # Process each y-slice (latitude) to exactly match emod3d.py write pattern
    with (
        open(vp3dfile, "wb") as fvp,
        open(vs3dfile, "wb") as fvs,
        open(rho3dfile, "wb") as frho,
        open(in_basin_mask_file, "wb") as fmask,
    ):
        for j in tqdm(range(ny), desc="Processing y-slices", unit="slice"):
            # Extract slice for this y-index from HDF5: (nz, ny, nx) -> (nz, nx)
            # The hdf5.py already stored the data transposed, so this gives us
            # exactly what emod3d.py produces with .T.flatten()
            vp_slice = vp_full[:, j, :]  # Shape: (nz, nx)
            vs_slice = vs_full[:, j, :]  # Shape: (nz, nx)
            rho_slice = rho_full[:, j, :]  # Shape: (nz, nx)
            inb_slice = inbasin_full[:, j, :]  # Shape: (nz, nx)

            # Apply minimum Vs constraint (match emod3d.py exactly)
            vs_slice = np.maximum(vs_slice, min_vs)

            # The HDF5 slice already has the correct layout from hdf5.py
            # Just flatten directly - no need for additional transposing
            vp_flat = vp_slice.flatten()  # (nz, nx) -> flat
            vs_flat = vs_slice.flatten()  # (nz, nx) -> flat
            rho_flat = rho_slice.flatten()  # (nz, nx) -> flat
            inb_flat = inb_slice.flatten()  # (nz, nx) -> flat

            # Convert inbasin to the same dtype as used in emod3d.py
            # emod3d.py: inbasin.astype(vp.dtype) where vp.dtype is float32
            inb_flat = inb_flat.astype(np.float32)

            # Debug: print first few values for the first slice
            if j == 0:
                print("\nFirst slice debug - first 10 values:")
                print(f"  vp:  {vp_flat[:10]}")
                print(f"  vs:  {vs_flat[:10]}")
                print(f"  rho: {rho_flat[:10]}")

            # Pack binary data with appropriate endianness (match emod3d.py exactly)
            vp_data = struct.pack(f"{endian_format}{len(vp_flat)}f", *vp_flat)
            vs_data = struct.pack(f"{endian_format}{len(vs_flat)}f", *vs_flat)
            rho_data = struct.pack(f"{endian_format}{len(rho_flat)}f", *rho_flat)
            inb_data = struct.pack(f"{endian_format}{len(inb_flat)}f", *inb_flat)

            # Write to binary files (append mode matches emod3d.py)
            fvp.write(vp_data)
            fvs.write(vs_data)
            frho.write(rho_data)
            fmask.write(inb_data)

    end_time = time.time()
    processing_time = end_time - start_time

    print("✅ Conversion complete!")
    print(f"   Created: {vp3dfile}")
    print(f"   Created: {vs3dfile}")
    print(f"   Created: {rho3dfile}")
    print(f"   Created: {in_basin_mask_file}")
    print(f"   Processing time: {processing_time:.2f} seconds")


@cli.from_docstring(app)
def convert_hdf5_to_emod3d_main(
        src_h5: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Path to input HDF5 file")],
        out_dir: Annotated[Path, typer.Argument(help="Output directory for binary files")],
        min_vs: Annotated[Optional[float], typer.Option(
            help="Minimum Vs value to enforce (km/s). If not specified, uses value from HDF5 metadata.")] = None,
) -> None:
    """
    Convert HDF5 velocity model to EMOD3D binary format.

    Converts HDF5 velocity model files (produced by hdf5.py) to the binary
    format expected by EMOD3D, producing the same output files as emod3d.py:
    - vp3dfile.p (P-wave velocities)
    - vs3dfile.s (S-wave velocities)
    - rho3dfile.d (densities)
    - in_basin_mask.b (basin membership mask)
    """
    try:
        convert_hdf5_to_emod3d(src_h5, out_dir, min_vs)
    except (FileNotFoundError, OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"❌ Error: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()