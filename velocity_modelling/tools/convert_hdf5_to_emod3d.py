#!/usr/bin/env python
# tools/convert_hdf5_to_emod3d.py
from pathlib import Path
import argparse, h5py, numpy as np

def main(src_h5: Path, out_dir: Path, zero_pad: int = 5):
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_h5, "r") as f:
        vp  = f["/properties/vp"]          # (nz, nx, ny)
        vs  = f["/properties/vs"]
        rho = f["/properties/rho"]
        inb = f["/properties/inbasin"]
        nz, nx, ny = vp.shape

        for j in range(ny):
            # Extract (nz, nx) then transpose to (nx, nz) to match your EMOD3D writer
            vp_j  = np.asarray(vp[:,  :, j]).T
            vs_j  = np.asarray(vs[:,  :, j]).T
            rho_j = np.asarray(rho[:, :, j]).T
            inb_j = np.asarray(inb[:, :, j]).T

            path = out_dir / f"slice_{j:0{zero_pad}d}.txt"
            with path.open("w") as fp:
                fp.write(f"# slice {j}\n# nx={nx} nz={nz}\n")
                # Example row format; mirror emod3d.py if needed
                for x in range(nx):
                    row = " ".join(f"{vp_j[x,z]:.6g},{vs_j[x,z]:.6g},{rho_j[x,z]:.6g},{int(inb_j[x,z])}"
                                   for z in range(nz))
                    fp.write(row + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src_h5", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--zero-pad", type=int, default=5)
    args = ap.parse_args()
    main(args.src_h5, args.out_dir, args.zero_pad)
