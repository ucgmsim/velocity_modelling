#!/usr/bin/env python
"""
Efficient HDF5 regression comparison for large files.

This script provides efficient methods for comparing large HDF5 files that should be identical
(regression testing). It removes unnecessary transpose complexity while keeping the fast
comparison methods for handling very large files.

Usage examples:
    # Statistical sampling (recommended) - samples 10,000 random points
    python compare_hdf5_regression.py file1.h5 file2.h5 --method sample --n-samples 10000

    # Systematic slice checking - checks 50 complete slices
    python compare_hdf5_regression.py file1.h5 file2.h5 --method slices --n-slices 50

    # Hash comparison - very fast, compares MD5 hashes of chunks
    python compare_hdf5_regression.py file1.h5 file2.h5 --method hash
"""

import hashlib
import time
from pathlib import Path
from typing import Annotated, Any

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


def stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate statistics for differences between arrays.

    Parameters
    ----------
    a : np.ndarray
        First array for comparison.
    b : np.ndarray
        Second array for comparison.

    Returns
    -------
    tuple[float, float, float]
        Tuple containing (mean_absolute_error, max_absolute_error, root_mean_square_error).
    """
    d = a - b
    mae = float(np.mean(np.abs(d)))
    mx = float(np.max(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    return mae, mx, rmse


def sample_comparison(
    ds_a: h5py.Dataset,
    ds_b: h5py.Dataset,
    n_samples: int = 10000,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    seed: int = 42,
) -> tuple[bool, str]:
    """
    Compare datasets using statistical sampling of random points.

    This method provides a statistically representative comparison by sampling
    random points throughout the dataset. It's efficient for large datasets
    while providing good confidence in the comparison results.

    Parameters
    ----------
    ds_a : h5py.Dataset
        First dataset to compare.
    ds_b : h5py.Dataset
        Second dataset to compare.
    n_samples : int, optional
        Number of random samples to compare, by default 10000.
    atol : float, optional
        Absolute tolerance for comparison, by default 1e-6.
    rtol : float, optional
        Relative tolerance for comparison, by default 1e-6.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    tuple[bool, str]
        Tuple of (datasets_match, statistics_string) where datasets_match
        indicates if the sampled points are within tolerance and statistics_string
        contains comparison metrics.

    Raises
    ------
    ValueError
        If datasets have incompatible shapes or dtypes.
    """
    if ds_a.shape != ds_b.shape:
        return False, f"shape mismatch: {ds_a.shape} vs {ds_b.shape}"

    if ds_a.dtype != ds_b.dtype:
        return False, f"dtype mismatch: {ds_a.dtype} vs {ds_b.dtype}"

    np.random.seed(seed)
    shape = ds_a.shape

    print(f"  Sampling {n_samples} random points from {shape} array...")

    # Generate random indices for the actual storage shape
    indices = []
    for _ in range(n_samples):
        idx = tuple(np.random.randint(0, dim) for dim in shape)
        indices.append(idx)

    # Sample from both datasets using identical indices
    sample_a = []
    sample_b = []

    for i, idx in enumerate(indices):
        sample_a.append(ds_a[idx])
        sample_b.append(ds_b[idx])

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
    n_slices: int = 50,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> tuple[bool, str]:
    """
    Check a systematic sample of complete slices along the first dimension.

    This method provides thorough checking by comparing complete slices
    systematically distributed throughout the dataset. It's more comprehensive
    than sampling but still efficient for large datasets.

    Parameters
    ----------
    ds_a : h5py.Dataset
        First dataset to compare.
    ds_b : h5py.Dataset
        Second dataset to compare.
    n_slices : int, optional
        Number of slices to check, by default 50.
    atol : float, optional
        Absolute tolerance for comparison, by default 1e-6.
    rtol : float, optional
        Relative tolerance for comparison, by default 1e-6.

    Returns
    -------
    tuple[bool, str]
        Tuple of (all_slices_match, summary_string) where all_slices_match
        indicates if all checked slices are within tolerance and summary_string
        contains details about any differences found.

    Raises
    ------
    ValueError
        If datasets have incompatible shapes or dtypes.
    """
    if ds_a.shape != ds_b.shape:
        return False, f"shape mismatch: {ds_a.shape} vs {ds_b.shape}"

    if ds_a.dtype != ds_b.dtype:
        return False, f"dtype mismatch: {ds_a.dtype} vs {ds_b.dtype}"

    shape = ds_a.shape
    first_dim = shape[0]
    slice_indices = np.linspace(0, first_dim - 1, n_slices, dtype=int)

    print(f"  Checking {n_slices} systematic slices: {list(slice_indices)}")

    differences = []

    for i, slice_idx in enumerate(slice_indices):
        # Load corresponding slices from both datasets
        slice_a = ds_a[slice_idx]
        slice_b = ds_b[slice_idx]

        # Compare slices
        if not np.allclose(slice_a, slice_b, atol=atol, rtol=rtol, equal_nan=True):
            slice_stats = stats(slice_a, slice_b)
            differences.append((slice_idx, slice_stats))

        status = "OK" if len(differences) == i else "DIFF"
        print(f"    Slice {slice_idx:4d}: {status}")

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
    chunk_slices: int = 100,
) -> tuple[bool, str]:
    """
    Compare using MD5 hash of chunks along the first dimension.

    This method provides the fastest comparison by computing MD5 hashes
    of data chunks. It can detect any differences but doesn't provide
    quantitative information about the magnitude of differences.

    Parameters
    ----------
    ds_a : h5py.Dataset
        First dataset to compare.
    ds_b : h5py.Dataset
        Second dataset to compare.
    chunk_slices : int, optional
        Number of slices per chunk for hashing, by default 100.

    Returns
    -------
    tuple[bool, str]
        Tuple of (hashes_match, summary_string) where hashes_match indicates
        if all chunk hashes are identical and summary_string contains details
        about any differing chunks.

    Raises
    ------
    ValueError
        If datasets have incompatible shapes or dtypes.
    """
    if ds_a.shape != ds_b.shape:
        return False, f"shape mismatch: {ds_a.shape} vs {ds_b.shape}"

    if ds_a.dtype != ds_b.dtype:
        return False, f"dtype mismatch: {ds_a.dtype} vs {ds_b.dtype}"

    shape = ds_a.shape
    first_dim = shape[0]

    print(f"  Computing hashes for chunks of {chunk_slices} slices...")

    hash_diffs = []

    for start in range(0, first_dim, chunk_slices):
        end = min(start + chunk_slices, first_dim)

        # Load chunks from both datasets
        chunk_a = ds_a[start:end]
        chunk_b = ds_b[start:end]

        # Compute hashes
        hash_a = hashlib.md5(chunk_a.tobytes()).hexdigest()
        hash_b = hashlib.md5(chunk_b.tobytes()).hexdigest()

        if hash_a != hash_b:
            hash_diffs.append((start, end))

        status = "SAME" if hash_a == hash_b else "DIFF"
        print(f"    Chunk {start:4d}-{end - 1:4d}: {status}")

    if hash_diffs:
        return False, f"{len(hash_diffs)} chunks have different hashes: {hash_diffs}"
    else:
        return True, "all chunk hashes match"


def compare_datasets(
    a_path: Path,
    b_path: Path,
    method: str = "sample",
    atol: float = 1e-6,
    rtol: float = 1e-6,
    n_samples: int = 10000,
    n_slices: int = 50,
    chunk_slices: int = 100,
) -> list[tuple[str, str, Any]]:
    """
    Compare HDF5 datasets using specified method.

    This function orchestrates the comparison of all standard velocity model
    datasets in two HDF5 files using the specified comparison method. It also
    compares configuration metadata.

    Parameters
    ----------
    a_path : Path
        Path to first HDF5 file.
    b_path : Path
        Path to second HDF5 file.
    method : str, optional
        Comparison method: 'sample', 'slices', or 'hash', by default "sample".
    atol : float, optional
        Absolute tolerance for comparison, by default 1e-6.
    rtol : float, optional
        Relative tolerance for comparison, by default 1e-6.
    n_samples : int, optional
        Number of samples for sample method, by default 10000.
    n_slices : int, optional
        Number of slices for slice method, by default 50.
    chunk_slices : int, optional
        Slices per chunk for hash method, by default 100.

    Returns
    -------
    list[tuple[str, str, Any]]
        List of (dataset_name, status, info) tuples where:
        - dataset_name: Name of the dataset or 'config'
        - status: 'identical', 'differs', 'missing', or 'error'
        - info: Details about the comparison result or error message

    Raises
    ------
    ValueError
        If an invalid comparison method is specified.
    FileNotFoundError
        If either HDF5 file cannot be opened.
    """
    methods = {
        "sample": sample_comparison,
        "slices": systematic_slice_check,
        "hash": hash_comparison,
    }

    if method not in methods:
        raise ValueError(f"Method must be one of: {list(methods.keys())}")

    results = []

    with h5py.File(a_path, "r") as file_a, h5py.File(b_path, "r") as file_b:
        for ds_name in DATASETS:
            print(f"\nComparing {ds_name} using {method} method...")
            start_time = time.time()

            if ds_name not in file_a or ds_name not in file_b:
                results.append((ds_name, "missing", None))
                continue

            ds_a = file_a[ds_name]
            ds_b = file_b[ds_name]

            try:
                # Apply chosen method
                if method == "sample":
                    is_match, info = sample_comparison(
                        ds_a, ds_b, n_samples, atol, rtol
                    )
                elif method == "slices":
                    is_match, info = systematic_slice_check(
                        ds_a, ds_b, n_slices, atol, rtol
                    )
                elif method == "hash":
                    is_match, info = hash_comparison(ds_a, ds_b, chunk_slices)

                elapsed = time.time() - start_time
                status = "identical" if is_match else "differs"
                results.append((ds_name, status, f"{info} ({elapsed:.1f}s)"))

            except (KeyError, ValueError, OSError) as e:
                results.append((ds_name, "error", str(e)))

        # Config comparison
        config_a = file_a.get("config")
        config_b = file_b.get("config")

        if config_a and config_b:
            config_diffs = []
            keys_a, keys_b = set(config_a.attrs), set(config_b.attrs)

            # Compare keys
            if keys_a != keys_b:
                missing_a = keys_b - keys_a
                missing_b = keys_a - keys_b
                if missing_a:
                    config_diffs.append(f"missing in A: {missing_a}")
                if missing_b:
                    config_diffs.append(f"missing in B: {missing_b}")

            # Compare values
            for key in keys_a & keys_b:
                val_a, val_b = config_a.attrs[key], config_b.attrs[key]
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    if not np.isclose(val_a, val_b, atol=atol, rtol=rtol):
                        config_diffs.append(f"{key}: {val_a} vs {val_b}")
                else:
                    if str(val_a) != str(val_b):
                        if key == "config_string":
                            config_diffs.append(f"{key}:\n\n{val_a}\n\nvs\n\n{val_b}")
                        else:
                            config_diffs.append(f"{key}: {val_a} vs {val_b}")

            status = "identical" if not config_diffs else "differs"
            results.append(
                (
                    "config",
                    status,
                    config_diffs if config_diffs else "all attributes match",
                )
            )
        else:
            results.append(
                ("config", "missing", "config group not found in one or both files")
            )

    return results


@cli.from_docstring(app)
def compare_hdf5_regression(
    a: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    b: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    method: str = "sample",
    atol: float = 1e-6,
    rtol: float = 1e-6,
    n_samples: int = 10000,
    n_slices: int = 50,
    chunk_slices: int = 100,
) -> None:
    """
    Efficient regression test comparison of large HDF5 velocity model files.

    This tool provides three efficient methods for comparing large velocity model
    files that should be identical (for regression testing):

    - **sample**: Statistical sampling of random points (recommended for most cases)
      Samples n_samples random points throughout each dataset and compares them.
      Fast and statistically representative.

    - **slices**: Systematic checking of complete slices (thorough)
      Checks n_slices complete slices systematically distributed through the dataset.
      More comprehensive than sampling but still efficient.

    - **hash**: MD5 hash comparison of chunks (fastest, detects any differences)
      Computes MD5 hashes of data chunks and compares them.
      Fastest method that can detect any differences, but doesn't quantify them.

    The tool compares standard velocity model datasets (/properties/vp, /properties/vs,
    /properties/rho, /properties/inbasin) and configuration metadata. It exits with
    code 0 if data is identical and code 1 if differences are found.

    Parameters
    ----------
    a : Path
        Path to the first HDF5 file to compare.
    b : Path
        Path to the second HDF5 file to compare.
    method : str, optional
        Comparison method to use, by default "sample".
    atol : float, optional
        Absolute tolerance for numerical comparisons, by default 1e-6.
    rtol : float, optional
        Relative tolerance for numerical comparisons, by default 1e-6.
    n_samples : int, optional
        Number of random samples to compare when using 'sample' method, by default 10000.
    n_slices : int, optional
        Number of slices to check when using 'slices' method, by default 50.
    chunk_slices : int, optional
        Number of slices per chunk when using 'hash' method, by default 100.

    Raises
    ------
    typer.BadParameter
        If an invalid comparison method is specified.
    typer.Exit
        Exits with code 1 if files differ, code 0 if identical.
    FileNotFoundError
        If either input file cannot be found or opened.

    Examples
    --------
    Compare two files using statistical sampling:

    >>> compare_hdf5_regression(Path("file1.h5"), Path("file2.h5"), method="sample", n_samples=5000)

    Compare using systematic slice checking:

    >>> compare_hdf5_regression(Path("file1.h5"), Path("file2.h5"), method="slices", n_slices=25)

    Quick hash-based comparison:

    >>> compare_hdf5_regression(Path("file1.h5"), Path("file2.h5"), method="hash")
    """
    # Validate method
    valid_methods = ["sample", "slices", "hash"]
    if method not in valid_methods:
        raise typer.BadParameter(f"Method must be one of: {valid_methods}")

    print(f"Efficient regression comparison: {a} vs {b}")
    print(f"Method: {method}")
    if method != "hash":
        print(f"Tolerances: atol={atol}, rtol={rtol}")

    start_total = time.time()
    results = compare_datasets(
        a, b, method, atol, rtol, n_samples, n_slices, chunk_slices
    )
    total_time = time.time() - start_total

    print(f"\n{'=' * 60}")
    print(f"REGRESSION TEST RESULTS ({total_time:.1f}s total):")
    print(f"{'=' * 60}")

    all_identical = True
    for ds, status, info in results:
        print(f"{ds:>22s} : {status}")
        if info and isinstance(info, str):
            print(f"{'':>25s}   {info}")
        elif info and isinstance(info, list):
            for item in info[:3]:  # Show first 3 differences
                print(f"{'':>25s}   {item}")
            if len(info) > 3:
                print(f"{'':>25s}   ... and {len(info) - 3} more differences")

        if status not in ["identical"]:
            all_identical = False

    # Check if data properties are identical (ignore config for pass/fail)
    data_identical = True
    for ds, status, info in results:
        if ds != "config" and status not in ["identical"]:
            data_identical = False
            break

    # Final result - only fail on data differences, not config differences
    if data_identical:
        result_msg = "REGRESSION TEST PASSED"
        if not all_identical:
            result_msg += " (config differs but data is identical)"
    else:
        result_msg = "REGRESSION TEST FAILED"

    print(f"\n{result_msg}")

    # Exit with appropriate code for CI/scripts
    if not all_identical:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
