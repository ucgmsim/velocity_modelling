from multiprocessing import Lock
from pathlib import Path
import h5py
import numpy as np

def convert_hdf5_to_emod3d_parallel(hdf5_file: Path, out_dir: Path):
    """
    Convert a HDF5-formatted velocity model into EMOD3D binary files in parallel (slices).

    Parameters
    ----------
    hdf5_file : Path
        Input HDF5 file path containing velocity model data.
    out_dir : Path
        Output directory to store EMOD3D binary files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_file, "r") as f:
        vp = f["/properties/vp"]
        vs = f["/properties/vs"]
        rho = f["/properties/rho"]
        inbasin = f["/properties/inbasin"]
        nz, ny, nx = vp.shape

        # Create empty binary files
        with open(out_dir / "vp3dfile.p", "wb") as fvp, \
             open(out_dir / "vs3dfile.s", "wb") as fvs, \
             open(out_dir / "rho3dfile.d", "wb") as frho, \
             open(out_dir / "in_basin_mask.b", "wb") as fbas:
            pass  # Just to create/truncate the files

    return {
        "vp_shape": (nz, ny, nx),
        "hdf5_file": hdf5_file,
        "out_dir": out_dir,
    }
