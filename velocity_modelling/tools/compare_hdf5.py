#!/usr/bin/env python
# tools/compare_hdf5.py
from pathlib import Path
import argparse, h5py, numpy as np

DATASETS = ["/properties/vp", "/properties/vs", "/properties/rho", "/properties/inbasin"]

def normalize(A: h5py.File, dsname: str):
    ds = A[dsname]
    shape = ds.shape
    cfg = A.get("config")
    if cfg is None:
        raise ValueError(f"{A.filename}: missing /config for axis mapping")
    nx = int(cfg.attrs["nx"]); ny = int(cfg.attrs["ny"]); nz = int(cfg.attrs["nz"])

    arr = np.asarray(ds[...])

    # If already (ny, nx, nz), done.
    if shape == (ny, nx, nz):
        return arr

    # Common serial layout (nz, nx, ny)
    if shape == (nz, nx, ny):
        # permute to (ny, nx, nz)
        return np.transpose(arr, (2,1,0))

    # Your earlier parallel layout (ny, nx, nz)
    if shape == (ny, nx, nz):
        return arr

    # Fallback: try to infer by matching unique dim 'nz'
    axes = list(shape)
    try:
        z_axis = axes.index(nz)
        # remaining two axes should be nx, ny (equal here, so we assume order (ny, nx, nz))
        if z_axis == 0:
            return np.transpose(arr, (2,1,0))
        elif z_axis == 1:
            return np.transpose(arr, (0,2,1))
        elif z_axis == 2:
            return arr
    except ValueError:
        pass
    raise ValueError(f"Unrecognized shape for {dsname}: {shape} with nx,ny,nz={nx,ny,nz}")

def stats(a, b):
    d = a - b
    mae = float(np.mean(np.abs(d)))
    mx  = float(np.max(np.abs(d)))
    rmse= float(np.sqrt(np.mean(d*d)))
    return mae, mx, rmse

def compare(a_path: Path, b_path: Path, atol=1e-6, rtol=1e-6, per_slice=False):
    out = []
    with h5py.File(a_path, "r") as A, h5py.File(b_path, "r") as B:
        for ds in DATASETS:
            if ds not in A or ds not in B:
                out.append((ds, "missing", None)); continue
            x = normalize(A, ds)  # -> (ny, nx, nz)
            y = normalize(B, ds)
            if x.shape != y.shape or x.dtype != y.dtype:
                out.append((ds, "shape/dtype mismatch", (x.shape, y.shape, str(x.dtype), str(y.dtype))))
                continue
            if per_slice:
                ny = x.shape[0]; diffs=[]
                for j in range(ny):
                    a = x[j]; b = y[j]
                    if not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                        diffs.append((j,)+stats(a,b))
                out.append((ds, "per-slice", diffs))
            else:
                ok = np.allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
                out.append((ds, "ok" if ok else "differs", None if ok else stats(x,y)))

        # compare selected config attrs
        CA, CB = A.get("config"), B.get("config")
        if CA and CB:
            keysA, keysB = set(CA.attrs), set(CB.attrs)
            # tolerate extra keys, only assert common ones
            common = sorted(keysA & keysB)
            skews=[]
            for k in common:
                va, vb = CA.attrs[k], CB.attrs[k]
                if isinstance(va,(int,float)) and isinstance(vb,(int,float)):
                    if not np.isclose(va, vb, atol=atol, rtol=rtol):
                        skews.append((k, va, vb))
                else:
                    if str(va) != str(vb):
                        skews.append((k, va, vb))
            out.append(("config", "ok" if not skews else "attr-values mismatch", skews))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--per-slice", action="store_true")
    args = ap.parse_args()
    res = compare(args.a, args.b, args.atol, args.rtol, args.per_slice)
    for ds, status, info in res:
        print(f"{ds:>22s} : {status}", "" if info is None else info)
