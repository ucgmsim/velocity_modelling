#!/usr/bin/env python
# tools/compare_hdf5_chunked.py
from pathlib import Path
import argparse, h5py, numpy as np

DATASETS = ["/properties/vp", "/properties/vs", "/properties/rho", "/properties/inbasin"]


def get_axis_mapping(cfg, shape):
    """Determine how to transpose data to (ny, nx, nz) format"""
    nx = int(cfg.attrs["nx"]);
    ny = int(cfg.attrs["ny"]);
    nz = int(cfg.attrs["nz"])

    # If already (ny, nx, nz), no transpose needed
    if shape == (ny, nx, nz):
        return None

    # Common serial layout (nz, nx, ny) -> (ny, nx, nz)
    if shape == (nz, nx, ny):
        return (2, 1, 0)

    # Try to infer by matching unique dim 'nz'
    axes = list(shape)
    try:
        z_axis = axes.index(nz)
        if z_axis == 0:  # (nz, ?, ?) -> (ny, nx, nz)
            return (2, 1, 0)
        elif z_axis == 1:  # (?, nz, ?) -> (ny, nx, nz)
            return (0, 2, 1)
        elif z_axis == 2:  # (?, ?, nz) -> already correct
            return None
    except ValueError:
        pass

    raise ValueError(f"Unrecognized shape: {shape} with nx,ny,nz={nx, ny, nz}")


def stats(a, b):
    """Calculate statistics for differences between arrays"""
    d = a - b
    mae = float(np.mean(np.abs(d)))
    mx = float(np.max(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    return mae, mx, rmse


def compare_chunked(ds_a, ds_b, transpose_a, transpose_b, atol=1e-6, rtol=1e-6, chunk_size=100):
    """Compare datasets chunk by chunk along the first axis"""
    shape_a = ds_a.shape
    shape_b = ds_b.shape

    # After transpose, shapes should match
    expected_shape_a = shape_a if transpose_a is None else tuple(shape_a[i] for i in transpose_a)
    expected_shape_b = shape_b if transpose_b is None else tuple(shape_b[i] for i in transpose_b)

    if expected_shape_a != expected_shape_b:
        return False, f"shape mismatch: {expected_shape_a} vs {expected_shape_b}"

    if ds_a.dtype != ds_b.dtype:
        return False, f"dtype mismatch: {ds_a.dtype} vs {ds_b.dtype}"

    # Process in chunks along first dimension
    first_dim = expected_shape_a[0]
    all_stats = []
    max_diff = 0

    for start in range(0, first_dim, chunk_size):
        end = min(start + chunk_size, first_dim)

        # Load chunks
        chunk_a = ds_a[start:end]
        chunk_b = ds_b[start:end]

        # Apply transpose if needed
        if transpose_a is not None:
            # Adjust transpose indices for the chunk (first dim is already selected)
            if transpose_a == (2, 1, 0):  # (nz, nx, ny) -> (ny, nx, nz), but we selected first dim
                # chunk shape is (chunk_size, nx, ny), we want (chunk_size, nx, ny) - no change needed
                pass
            elif transpose_a == (0, 2, 1):  # (ny, nz, nx) -> (ny, nx, nz)
                chunk_a = np.transpose(chunk_a, (0, 2, 1))

        if transpose_b is not None:
            if transpose_b == (2, 1, 0):
                pass
            elif transpose_b == (0, 2, 1):
                chunk_b = np.transpose(chunk_b, (0, 2, 1))

        # Compare chunks
        if not np.allclose(chunk_a, chunk_b, atol=atol, rtol=rtol, equal_nan=True):
            chunk_stats = stats(chunk_a, chunk_b)
            all_stats.append((start, end, chunk_stats))
            max_diff = max(max_diff, chunk_stats[1])  # max absolute diff

        print(f"  Processed slices {start:4d}-{end - 1:4d} / {first_dim - 1}")

    if all_stats:
        return False, f"differences found in {len(all_stats)} chunks, max_diff={max_diff:.2e}"
    else:
        return True, "chunks match"


def compare(a_path: Path, b_path: Path, atol=1e-6, rtol=1e-6, chunk_size=100):
    """Compare HDF5 files using chunked processing"""
    out = []

    with h5py.File(a_path, "r") as A, h5py.File(b_path, "r") as B:
        for ds_name in DATASETS:
            print(f"\nComparing {ds_name}...")

            if ds_name not in A or ds_name not in B:
                out.append((ds_name, "missing", None))
                continue

            ds_a = A[ds_name]
            ds_b = B[ds_name]

            # Get config for axis mapping
            cfg_a = A.get("config")
            cfg_b = B.get("config")

            if cfg_a is None or cfg_b is None:
                out.append((ds_name, "missing config", None))
                continue

            try:
                transpose_a = get_axis_mapping(cfg_a, ds_a.shape)
                transpose_b = get_axis_mapping(cfg_b, ds_b.shape)

                print(f"  Shape A: {ds_a.shape}, transpose: {transpose_a}")
                print(f"  Shape B: {ds_b.shape}, transpose: {transpose_b}")

                is_match, info = compare_chunked(ds_a, ds_b, transpose_a, transpose_b,
                                                 atol, rtol, chunk_size)

                status = "ok" if is_match else "differs"
                out.append((ds_name, status, info))

            except Exception as e:
                out.append((ds_name, "error", str(e)))

        # Compare config attributes
        print(f"\nComparing config attributes...")
        CA, CB = A.get("config"), B.get("config")
        if CA and CB:
            keysA, keysB = set(CA.attrs), set(CB.attrs)
            common = sorted(keysA & keysB)
            skews = []

            for k in common:
                va, vb = CA.attrs[k], CB.attrs[k]
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    if not np.isclose(va, vb, atol=atol, rtol=rtol):
                        skews.append((k, va, vb))
                else:
                    if str(va) != str(vb):
                        skews.append((k, va, vb))

            status = "ok" if not skews else "attr-values mismatch"
            out.append(("config", status, skews))

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare large HDF5 files using chunked processing")
    ap.add_argument("a", type=Path, help="First HDF5 file")
    ap.add_argument("b", type=Path, help="Second HDF5 file")
    ap.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    ap.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance")
    ap.add_argument("--chunk-size", type=int, default=100,
                    help="Number of slices to process at once (reduce if memory issues)")
    args = ap.parse_args()

    print(f"Comparing {args.a} vs {args.b}")
    print(f"Tolerances: atol={args.atol}, rtol={args.rtol}")
    print(f"Chunk size: {args.chunk_size} slices")

    res = compare(args.a, args.b, args.atol, args.rtol, args.chunk_size)

    print(f"\n{'=' * 60}")
    print("COMPARISON RESULTS:")
    print(f"{'=' * 60}")

    for ds, status, info in res:
        print(f"{ds:>22s} : {status}")
        if info and info != "chunks match":
            print(f"{'':>25s}   {info}")