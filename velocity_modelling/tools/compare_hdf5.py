#!/usr/bin/env python
"""
compare_hdf5_sampling.py: Fast comparison of HDF5 velocity model files.

This script provides several methods for comparing large HDF5 files containing seismic velocity models:
  - Statistical sampling (recommended): samples random points for efficient comparison.
  - Systematic slice checking: compares complete slices for thoroughness.
  - Hash comparison: compares MD5 hashes of chunks for very fast integrity checks.

Usage examples:
    # Statistical sampling (recommended) - samples 10,000 random points
    python compare_hdf5_sampling.py vm3_24Sep/velocity_model.h5 vm3_25Sep/velocity_model.h5 --method sample --n-samples 10000

    # Systematic slice checking - checks 50 complete slices
    python compare_hdf5_sampling.py vm3_24Sep/velocity_model.h5 vm3_25Sep/velocity_model.h5 --method slices --n-slices 50

    # Hash comparison - very fast, compares MD5 hashes of chunks
    python compare_hdf5_sampling.py vm3_24Sep/velocity_model.h5 vm3_25Sep/velocity_model.h5 --method hash

For more detailed comparison, consider using h5diff:
    # Basic comparison
    h5diff vm3_24Sep/velocity_model.h5 vm3_25Sep/velocity_model.h5

    # With tolerance
    h5diff -d 1e-6 vm3_24Sep/velocity_model.h5 vm3_25Sep/velocity_model.h5

    # Compare specific datasets
    h5diff -d 1e-6 vm3_24Sep/velocity_model.h5 vm3_25Sep/velocity_model.h5 /properties/vp /properties/vp
"""
from pathlib import Path
import argparse, h5py, numpy as np
import time

DATASETS = ["/properties/vp", "/properties/vs", "/properties/rho", "/properties/inbasin"]


def get_axis_mapping(cfg, shape):
    """Determine how to transpose data to (ny, nx, nz) format"""
    nx = int(cfg.attrs["nx"]);
    ny = int(cfg.attrs["ny"]);
    nz = int(cfg.attrs["nz"])

    if shape == (ny, nx, nz):
        return None
    elif shape == (nz, nx, ny):
        return (2, 1, 0)
    elif shape == (ny, nz, nx):
        return (0, 2, 1)
    else:
        # Try to infer
        axes = list(shape)
        try:
            z_axis = axes.index(nz)
            if z_axis == 0:
                return (2, 1, 0)
            elif z_axis == 1:
                return (0, 2, 1)
            elif z_axis == 2:
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


def sample_comparison(ds_a, ds_b, transpose_a, transpose_b, target_shape,
                      n_samples=1000, atol=1e-6, rtol=1e-6, seed=42):
    """Compare datasets using statistical sampling"""

    if ds_a.dtype != ds_b.dtype:
        return False, f"dtype mismatch: {ds_a.dtype} vs {ds_b.dtype}"

    ny, nx, nz = target_shape
    np.random.seed(seed)

    print(f"  Sampling {n_samples} random points from {ny}×{nx}×{nz} array...")

    # Generate random indices
    y_indices = np.random.randint(0, ny, n_samples)
    x_indices = np.random.randint(0, nx, n_samples)
    z_indices = np.random.randint(0, nz, n_samples)

    # Sample from both datasets
    sample_a = []
    sample_b = []

    for i, (y, x, z) in enumerate(zip(y_indices, x_indices, z_indices)):
        # Convert to original indices based on transpose
        if transpose_a == (2, 1, 0):  # (nz, nx, ny) -> (ny, nx, nz)
            orig_a = (z, x, y)
        elif transpose_a == (0, 2, 1):  # (ny, nz, nx) -> (ny, nx, nz)
            orig_a = (y, z, x)
        else:  # None
            orig_a = (y, x, z)

        if transpose_b == (2, 1, 0):
            orig_b = (z, x, y)
        elif transpose_b == (0, 2, 1):
            orig_b = (y, z, x)
        else:
            orig_b = (y, x, z)

        val_a = ds_a[orig_a]
        val_b = ds_b[orig_b]

        sample_a.append(val_a)
        sample_b.append(val_b)

        if (i + 1) % 100 == 0:
            print(f"    Sampled {i + 1}/{n_samples} points...")

    sample_a = np.array(sample_a)
    sample_b = np.array(sample_b)

    # Compare samples
    matches = np.allclose(sample_a, sample_b, atol=atol, rtol=rtol, equal_nan=True)
    sample_stats = stats(sample_a, sample_b)

    return matches, f"sample_stats: MAE={sample_stats[0]:.2e}, Max={sample_stats[1]:.2e}, RMSE={sample_stats[2]:.2e}"


def systematic_slice_check(ds_a, ds_b, transpose_a, transpose_b, target_shape,
                           n_slices=20, atol=1e-6, rtol=1e-6):
    """Check a systematic sample of complete slices"""

    ny, nx, nz = target_shape
    slice_indices = np.linspace(0, ny - 1, n_slices, dtype=int)

    print(f"  Checking {n_slices} systematic slices: {list(slice_indices)}")

    differences = []

    for i, slice_idx in enumerate(slice_indices):
        # Load slice
        if transpose_a == (2, 1, 0):  # (nz, nx, ny)
            slice_a = ds_a[:, :, slice_idx]  # (nz, nx)
            slice_a = slice_a.T  # -> (nx, nz)
        elif transpose_a == (0, 2, 1):  # (ny, nz, nx)
            slice_a = ds_a[slice_idx, :, :]  # (nz, nx)
            slice_a = slice_a.T  # -> (nx, nz)
        else:  # (ny, nx, nz)
            slice_a = ds_a[slice_idx, :, :]  # (nx, nz)

        if transpose_b == (2, 1, 0):
            slice_b = ds_b[:, :, slice_idx]
            slice_b = slice_b.T
        elif transpose_b == (0, 2, 1):
            slice_b = ds_b[slice_idx, :, :]
            slice_b = slice_b.T
        else:
            slice_b = ds_b[slice_idx, :, :]

        # Compare slice
        if not np.allclose(slice_a, slice_b, atol=atol, rtol=rtol, equal_nan=True):
            slice_stats = stats(slice_a, slice_b)
            differences.append((slice_idx, slice_stats))

        print(f"    Slice {slice_idx:4d}: {'OK' if len(differences) == i else 'DIFF'}")

    if differences:
        max_diff = max(diff[1][1] for diff in differences)  # max of max diffs
        return False, f"{len(differences)}/{n_slices} slices differ, max_diff={max_diff:.2e}"
    else:
        return True, f"all {n_slices} slices match"


def hash_comparison(ds_a, ds_b, transpose_a, transpose_b, target_shape, chunk_slices=100):
    """Compare using hash/checksum of chunks"""
    import hashlib

    ny, nx, nz = target_shape

    print(f"  Computing hashes for chunks of {chunk_slices} slices...")

    hash_diffs = []

    for start in range(0, ny, chunk_slices):
        end = min(start + chunk_slices, ny)

        # Load chunks
        if transpose_a == (2, 1, 0):  # (nz, nx, ny)
            chunk_a = ds_a[:, :, start:end]
            chunk_a = np.transpose(chunk_a, (2, 1, 0))  # -> (chunk_size, nx, nz)
        else:
            chunk_a = ds_a[start:end]
            if transpose_a: chunk_a = np.transpose(chunk_a, transpose_a)

        if transpose_b == (2, 1, 0):
            chunk_b = ds_b[:, :, start:end]
            chunk_b = np.transpose(chunk_b, (2, 1, 0))
        else:
            chunk_b = ds_b[start:end]
            if transpose_b: chunk_b = np.transpose(chunk_b, transpose_b)

        # Compute hashes
        hash_a = hashlib.md5(chunk_a.tobytes()).hexdigest()
        hash_b = hashlib.md5(chunk_b.tobytes()).hexdigest()

        if hash_a != hash_b:
            hash_diffs.append((start, end))

        print(f"    Chunk {start:4d}-{end - 1:4d}: {'SAME' if hash_a == hash_b else 'DIFF'}")

    if hash_diffs:
        return False, f"{len(hash_diffs)} chunks have different hashes: {hash_diffs}"
    else:
        return True, "all chunk hashes match"


def compare_fast(a_path: Path, b_path: Path, atol=1e-6, rtol=1e-6,
                 method="sample", n_samples=1000, n_slices=20):
    """Fast comparison using various methods"""

    methods = {
        "sample": sample_comparison,
        "slices": systematic_slice_check,
        "hash": hash_comparison
    }

    if method not in methods:
        raise ValueError(f"Method must be one of: {list(methods.keys())}")

    out = []

    with h5py.File(a_path, "r") as A, h5py.File(b_path, "r") as B:
        for ds_name in DATASETS:
            print(f"\nComparing {ds_name} using {method} method...")
            start_time = time.time()

            if ds_name not in A or ds_name not in B:
                out.append((ds_name, "missing", None))
                continue

            ds_a = A[ds_name]
            ds_b = B[ds_name]

            cfg_a = A.get("config")
            cfg_b = B.get("config")

            if cfg_a is None or cfg_b is None:
                out.append((ds_name, "missing config", None))
                continue

            try:
                transpose_a = get_axis_mapping(cfg_a, ds_a.shape)
                transpose_b = get_axis_mapping(cfg_b, ds_b.shape)

                # Determine target shape after transpose
                nx = int(cfg_a.attrs["nx"]);
                ny = int(cfg_a.attrs["ny"]);
                nz = int(cfg_a.attrs["nz"])
                target_shape = (ny, nx, nz)

                print(f"  Shape A: {ds_a.shape}, transpose: {transpose_a}")
                print(f"  Shape B: {ds_b.shape}, transpose: {transpose_b}")

                # Apply chosen method
                if method == "sample":
                    is_match, info = sample_comparison(ds_a, ds_b, transpose_a, transpose_b,
                                                       target_shape, n_samples, atol, rtol)
                elif method == "slices":
                    is_match, info = systematic_slice_check(ds_a, ds_b, transpose_a, transpose_b,
                                                            target_shape, n_slices, atol, rtol)
                elif method == "hash":
                    is_match, info = hash_comparison(ds_a, ds_b, transpose_a, transpose_b, target_shape)

                elapsed = time.time() - start_time
                status = "ok" if is_match else "differs"
                out.append((ds_name, status, f"{info} ({elapsed:.1f}s)"))

            except Exception as e:
                out.append((ds_name, "error", str(e)))

        # Config comparison (fast)
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
            out.append(("config", "ok" if not skews else "differs", skews))

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fast HDF5 comparison using sampling/hashing")
    ap.add_argument("a", type=Path, help="First HDF5 file")
    ap.add_argument("b", type=Path, help="Second HDF5 file")
    ap.add_argument("--method", choices=["sample", "slices", "hash"], default="sample",
                    help="Comparison method: sample (random points), slices (systematic slices), hash (chunk hashes)")
    ap.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    ap.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance")
    ap.add_argument("--n-samples", type=int, default=10000, help="Number of random samples (for sample method)")
    ap.add_argument("--n-slices", type=int, default=50, help="Number of slices to check (for slices method)")
    args = ap.parse_args()

    print(f"Fast comparing {args.a} vs {args.b}")
    print(f"Method: {args.method}")
    print(f"Tolerances: atol={args.atol}, rtol={args.rtol}")

    start_total = time.time()
    res = compare_fast(args.a, args.b, args.atol, args.rtol, args.method, args.n_samples, args.n_slices)
    total_time = time.time() - start_total

    print(f"\n{'=' * 60}")
    print(f"COMPARISON RESULTS ({total_time:.1f}s total):")
    print(f"{'=' * 60}")

    for ds, status, info in res:
        print(f"{ds:>22s} : {status}")
        if info and isinstance(info, str):
            print(f"{'':>25s}   {info}")
        elif info:
            print(f"{'':>25s}   {info}")