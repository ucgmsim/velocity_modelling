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

import time
from pathlib import Path
from typing import Annotated, Any, Optional

import h5py
import numpy as np
import typer
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)

DATASETS = [
    "/properties/vp",
    "/properties/vs",
    "/properties/rho",
    "/properties/inbasin",
]


def get_axis_mapping(
        cfg: h5py.Group, shape: tuple[int, ...]
) -> Optional[tuple[int, ...]]:
    """
    Determine how to transpose data to (ny, nx, nz) format.

    Parameters
    ----------
    cfg : h5py.Group
        HDF5 config group containing nx, ny, nz attributes.
    shape : tuple[int, ...]
        Current shape of the dataset.

    Returns
    -------
    tuple[int, ...] or None
        Transpose indices to convert to (ny, nx, nz), or None if no transpose needed.
    """
    nx = int(cfg.attrs["nx"])
    ny = int(cfg.attrs["ny"])
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


def stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate statistics for differences between arrays.

    Parameters
    ----------
    a : np.ndarray
        First array.
    b : np.ndarray
        Second array.

    Returns
    -------
    tuple[float, float, float]
        Tuple of (MAE, max absolute difference, RMSE).
    """
    d = a - b
    mae = float(np.mean(np.abs(d)))
    mx = float(np.max(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    return mae, mx, rmse


def sample_comparison(
        ds_a: h5py.Dataset,
        ds_b: h5py.Dataset,
        transpose_a: Optional[tuple[int, ...]],
        transpose_b: Optional[tuple[int, ...]],
        target_shape: tuple[int, int, int],
        n_samples: int = 1000,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        seed: int = 42,
) -> tuple[bool, str]:
    """
    Compare datasets using statistical sampling.

    Parameters
    ----------
    ds_a : h5py.Dataset
        First dataset to compare.
    ds_b : h5py.Dataset
        Second dataset to compare.
    transpose_a : tuple[int, ...] or None
        Transpose indices for first dataset.
    transpose_b : tuple[int, ...] or None
        Transpose indices for second dataset.
    target_shape : tuple[int, int, int]
        Target shape (ny, nx, nz).
    n_samples : int
        Number of random samples to compare.
    atol : float
        Absolute tolerance for comparison.
    rtol : float
        Relative tolerance for comparison.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[bool, str]
        Tuple of (matches, info_string).
    """
    if ds_a.dtype != ds_b.dtype:
        return False, f"dtype mismatch: {ds_a.dtype} vs {ds_b.dtype}"

    # Use original shapes for index generation
    orig_shape_a = ds_a.shape
    orig_shape_b = ds_b.shape

    ny, nx, nz = target_shape
    np.random.seed(seed)

    print(f"  Sampling {n_samples} random points from {ny}×{nx}×{nz} array...")
    print(f"  Original shapes: A={orig_shape_a}, B={orig_shape_b}")

    # Generate random indices for ORIGINAL shapes
    indices_a = []
    indices_b = []

    for _ in range(n_samples):
        # Generate random indices for original array dimensions
        idx_a = tuple(np.random.randint(0, dim) for dim in orig_shape_a)
        idx_b = tuple(np.random.randint(0, dim) for dim in orig_shape_b)
        indices_a.append(idx_a)
        indices_b.append(idx_b)

    # Sample from both datasets
    sample_a = []
    sample_b = []

    for i, (idx_a, idx_b) in enumerate(zip(indices_a, indices_b)):
        val_a = ds_a[idx_a]
        val_b = ds_b[idx_b]

        sample_a.append(val_a)
        sample_b.append(val_b)

        if (i + 1) % 1000 == 0:
            print(f"    Sampled {i + 1}/{n_samples} points...")

    sample_a = np.array(sample_a)
    sample_b = np.array(sample_b)

    # Compare samples
    matches = np.allclose(sample_a, sample_b, atol=atol, rtol=rtol, equal_nan=True)
    sample_stats = stats(sample_a, sample_b)

    return (
        matches,
        f"sample_stats: MAE={sample_stats[0]:.2e}, Max={sample_stats[1]:.2e}, RMSE={sample_stats[2]:.2e}",
    )


def systematic_slice_check(
        ds_a: h5py.Dataset,
        ds_b: h5py.Dataset,
        transpose_a: tuple[int, ...] | None,
        transpose_b: tuple[int, ...] | None,
        target_shape: tuple[int, int, int],
        n_slices: int = 20,
        atol: float = 1e-6,
        rtol: float = 1e-6,
) -> tuple[bool, str]:
    """
    Check a systematic sample of complete slices.

    Parameters
    ----------
    ds_a : h5py.Dataset
        First dataset to compare.
    ds_b : h5py.Dataset
        Second dataset to compare.
    transpose_a : tuple[int, ...] or None
        Transpose indices for first dataset.
    transpose_b : tuple[int, ...] or None
        Transpose indices for second dataset.
    target_shape : tuple[int, int, int]
        Target shape (ny, nx, nz).
    n_slices : int
        Number of slices to check.
    atol : float
        Absolute tolerance for comparison.
    rtol : float
        Relative tolerance for comparison.

    Returns
    -------
    tuple[bool, str]
        Tuple of (matches, info_string).
    """
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
        return (
            False,
            f"{len(differences)}/{n_slices} slices differ, max_diff={max_diff:.2e}",
        )
    else:
        return True, f"all {n_slices} slices match"


def hash_comparison(
        ds_a: h5py.Dataset,
        ds_b: h5py.Dataset,
        transpose_a: tuple[int, ...] | None,
        transpose_b: tuple[int, ...] | None,
        target_shape: tuple[int, int, int],
        chunk_slices: int = 100,
) -> tuple[bool, str]:
    """
    Compare using hash/checksum of chunks.

    Parameters
    ----------
    ds_a : h5py.Dataset
        First dataset to compare.
    ds_b : h5py.Dataset
        Second dataset to compare.
    transpose_a : tuple[int, ...] or None
        Transpose indices for first dataset.
    transpose_b : tuple[int, ...] or None
        Transpose indices for second dataset.
    target_shape : tuple[int, int, int]
        Target shape (ny, nx, nz).
    chunk_slices : int
        Number of slices per chunk for hashing.

    Returns
    -------
    tuple[bool, str]
        Tuple of (matches, info_string).
    """
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
            if transpose_a:
                chunk_a = np.transpose(chunk_a, transpose_a)

        if transpose_b == (2, 1, 0):
            chunk_b = ds_b[:, :, start:end]
            chunk_b = np.transpose(chunk_b, (2, 1, 0))
        else:
            chunk_b = ds_b[start:end]
            if transpose_b:
                chunk_b = np.transpose(chunk_b, transpose_b)

        # Compute hashes
        hash_a = hashlib.md5(chunk_a.tobytes()).hexdigest()
        hash_b = hashlib.md5(chunk_b.tobytes()).hexdigest()

        if hash_a != hash_b:
            hash_diffs.append((start, end))

        print(
            f"    Chunk {start:4d}-{end - 1:4d}: {'SAME' if hash_a == hash_b else 'DIFF'}"
        )

    if hash_diffs:
        return False, f"{len(hash_diffs)} chunks have different hashes: {hash_diffs}"
    else:
        return True, "all chunk hashes match"


def compare_fast(
        a_path: Path,
        b_path: Path,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        method: str = "sample",
        n_samples: int = 1000,
        n_slices: int = 20,
) -> list[tuple[str, str, Any]]:
    """
    Fast comparison using various methods.

    Parameters
    ----------
    a_path : Path
        Path to first HDF5 file.
    b_path : Path
        Path to second HDF5 file.
    atol : float
        Absolute tolerance for comparison.
    rtol : float
        Relative tolerance for comparison.
    method : str
        Comparison method: 'sample', 'slices', or 'hash'.
    n_samples : int
        Number of samples for sample method.
    n_slices : int
        Number of slices for slice method.

    Returns
    -------
    list[tuple[str, str, Any]]
        List of (dataset_name, status, info) tuples.
    """
    methods = {
        "sample": sample_comparison,
        "slices": systematic_slice_check,
        "hash": hash_comparison,
    }

    if method not in methods:
        raise ValueError(f"Method must be one of: {list(methods.keys())}")

    out = []

    with h5py.File(a_path, "r") as file_a, h5py.File(b_path, "r") as file_b:
        for ds_name in DATASETS:
            print(f"\nComparing {ds_name} using {method} method...")
            start_time = time.time()

            if ds_name not in file_a or ds_name not in file_b:
                out.append((ds_name, "missing", None))
                continue

            ds_a = file_a[ds_name]
            ds_b = file_b[ds_name]

            cfg_a = file_a.get("config")
            cfg_b = file_b.get("config")

            if cfg_a is None or cfg_b is None:
                out.append((ds_name, "missing config", None))
                continue

            try:
                transpose_a = get_axis_mapping(cfg_a, ds_a.shape)
                transpose_b = get_axis_mapping(cfg_b, ds_b.shape)

                # Determine target shape after transpose
                nx = int(cfg_a.attrs["nx"])
                ny = int(cfg_a.attrs["ny"])
                nz = int(cfg_a.attrs["nz"])
                target_shape = (ny, nx, nz)

                print(f"  Shape A: {ds_a.shape}, transpose: {transpose_a}")
                print(f"  Shape B: {ds_b.shape}, transpose: {transpose_b}")

                # Apply chosen method
                if method == "sample":
                    is_match, info = sample_comparison(
                        ds_a,
                        ds_b,
                        transpose_a,
                        transpose_b,
                        target_shape,
                        n_samples,
                        atol,
                        rtol,
                    )
                elif method == "slices":
                    is_match, info = systematic_slice_check(
                        ds_a,
                        ds_b,
                        transpose_a,
                        transpose_b,
                        target_shape,
                        n_slices,
                        atol,
                        rtol,
                    )
                elif method == "hash":
                    is_match, info = hash_comparison(
                        ds_a, ds_b, transpose_a, transpose_b, target_shape
                    )

                elapsed = time.time() - start_time
                status = "ok" if is_match else "differs"
                out.append((ds_name, status, f"{info} ({elapsed:.1f}s)"))

            except (KeyError, ValueError, OSError) as e:
                out.append((ds_name, "error", str(e)))

        # Config comparison (fast)
        config_a, config_b = file_a.get("config"), file_b.get("config")
        if config_a and config_b:
            keys_a, keys_b = set(config_a.attrs), set(config_b.attrs)
            common = sorted(keys_a & keys_b)
            skews = []
            for k in common:
                va, vb = config_a.attrs[k], config_b.attrs[k]
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    if not np.isclose(va, vb, atol=atol, rtol=rtol):
                        skews.append((k, va, vb))
                else:
                    if str(va) != str(vb):
                        skews.append((k, va, vb))
            out.append(("config", "ok" if not skews else "differs", skews))

    return out


@cli.from_docstring(app)
def compare_hdf5_sampling(
        a: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="First HDF5 file")],
        b: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Second HDF5 file")],
        method: Annotated[str, typer.Option(help="Comparison method")] = "sample",
        atol: Annotated[float, typer.Option(help="Absolute tolerance")] = 1e-6,
        rtol: Annotated[float, typer.Option(help="Relative tolerance")] = 1e-6,
        n_samples: Annotated[int, typer.Option(help="Number of random samples (for sample method)")] = 10000,
        n_slices: Annotated[int, typer.Option(help="Number of slices to check (for slices method)")] = 50,
) -> None:
    """
    Fast comparison of HDF5 velocity model files.

    Provides several methods for comparing large HDF5 files containing seismic velocity models:
      - Statistical sampling (recommended): samples random points for efficient comparison
      - Systematic slice checking: compares complete slices for thoroughness
      - Hash comparison: compares MD5 hashes of chunks for very fast integrity checks
    """
    # Validate method
    valid_methods = ["sample", "slices", "hash"]
    if method not in valid_methods:
        raise typer.BadParameter(f"Method must be one of: {valid_methods}")

    print(f"Fast comparing {a} vs {b}")
    print(f"Method: {method}")
    print(f"Tolerances: atol={atol}, rtol={rtol}")

    start_total = time.time()
    res = compare_fast(a, b, atol, rtol, method, n_samples, n_slices)
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


if __name__ == "__main__":
    app()