"""
Compare the output files from two directories containing the output files from emod3d.

This is still experimental

Usage:
    python compare_emod3d.py output_dir1 output_dir2 nzcvm.cfg

Example output:
```
Model dimensions: nx=100, ny=100, nz=225
Expected grid size (nx * ny * nz): 2250000
Size check for vp:
  Data1: 2250000 elements, matches expected size: True
  Data2: 2250000 elements, matches expected size: True
...
Comparing vp with length 2250000 Data1 len=2250000 Data2 len=2250000
Data1: max=8.906652450561523, min=1.7999999523162842 mean=6.716171741485596 std=1.10097074508667
Data2: max=8.906652450561523, min=1.7999999523162842 mean=6.716171741485596 std=1.10097074508667
...
Results for vp:
{'allclose': True, 'difference': array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), 'max_diff': 9.536743e-07,
'min_diff': 0.0, 'mean_diff': 3.390842e-12, 'std_diff': 1.4216468e-09, 'first_significant_index': None, 'x_index': None,
 'y_index': None, 'z_index': None, 'size_check': True}

...

Summary:
  vp: MATCH (max diff: 9.5367431640625e-07)
  vs: MATCH (max diff: 4.76837158203125e-07)
  rho: MATCH (max diff: 2.384185791015625e-07)
```

It only considers 'allclose' for vp, vs and rho. inbasin from the original C code is incorrect, so it is almost always False.

"""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from qcore import cli
from velocity_modelling.scripts.generate_3d_model import (
    parse_nzcvm_config,
)
from velocity_modelling.write.emod3d import (
    read_emomd3d_vm,
)

app = typer.Typer(pretty_exceptions_enable=False)


def compare_output_files(
    dir1: Path, dir2: Path, nx: int, ny: int, nz: int, threshold: float = 1e-5
):
    """
    Compare the output files from two directories containing the output files from emod3d.

    Parameters
    ----------
    dir1 : Path
        Directory name that contains the first set of emod3d output files.
    dir2 : Path
        Directory name that contains the second set of emod3 output files.
    nx : int
        Number of grid points in the x direction (along the longitude).
    ny : int
        Number of grid points in the y direction (along the latitude).
    nz : int
        Number of grid points in the z direction (along the depth).
    threshold : float, optional
        The threshold for the difference between the two files. The default is 1e-5.

    Returns
    -------
    dict
        Dictionary contains the comparison results for each key in the data.
    """
    data1 = read_emomd3d_vm(dir1)
    data2 = read_emomd3d_vm(dir2)

    # Calculate expected grid size
    expected_size = nx * ny * nz
    print(f"Expected grid size (nx * ny * nz): {expected_size}")

    # Check if data sizes match expected grid size
    size_checks = {}
    for key in data1:
        data1_size = len(data1[key])
        size_check_1 = data1_size == expected_size

        data2_size = len(data2.get(key, []))
        size_check_2 = data2_size == expected_size

        size_checks[key] = data1_size == data2_size == expected_size

        print(f"Size check for {key}:")
        print(f"  Data1: {data1_size} elements, matches expected size: {size_check_1}")
        print(f"  Data2: {data2_size} elements, matches expected size: {size_check_2}")

    comparison = {}
    for key in data1:
        if key in data2:
            min_length = min(len(data1[key]), len(data2[key]))
            print(
                f"Comparing {key} with length {min_length} Data1 len={len(data1[key])} Data2 len={len(data2[key])}"
            )
            data1_trimmed = data1[key][:min_length]
            data2_trimmed = data2[key][:min_length]
            print(
                f"Data1: max={np.max(data1_trimmed)}, min={np.min(data1_trimmed)} mean={np.mean(data1_trimmed)} std={np.std(data1_trimmed)}"
            )
            print(
                f"Data2: max={np.max(data2_trimmed)}, min={np.min(data2_trimmed)} mean={np.mean(data2_trimmed)} std={np.std(data2_trimmed)}"
            )
            print()  # for better readability

            difference = data1_trimmed - data2_trimmed

            significant_diff_indices = np.where(np.abs(difference) > threshold)[0]

            if significant_diff_indices.size > 0:
                first_significant_index = significant_diff_indices[0]
                x_index = first_significant_index % nx
                z_index = (first_significant_index // nx) % nz
                y_index = first_significant_index // (nx * nz)

            else:
                x_index = y_index = z_index = first_significant_index = None

            comparison[key] = {
                "allclose": np.allclose(data1_trimmed, data2_trimmed),
                "difference": np.abs(difference),
                "max_diff": np.max(np.abs(difference)),
                "min_diff": np.min(np.abs(difference)),
                "mean_diff": np.mean(np.abs(difference)),
                "std_diff": np.std(np.abs(difference)),
                "first_significant_index": first_significant_index,
                "x_index": x_index,
                "y_index": y_index,
                "z_index": z_index,
                "size_check": size_checks[key],
            }

        else:
            comparison[key] = "File missing in second directory"

    return comparison


@cli.from_docstring(app)
def compare_emod3d_dirs(
    output_dir1: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir2: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    nzcvm_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    threshold: Annotated[float, typer.Option(help="Difference threshold")] = 1e-5,
) -> None:
    """
    Compare output files from two EMOD3D velocity model directories.

    This tool compares velocity model files (vp, vs, rho, inbasin) from two
    directories and reports statistical differences and alignment.

    Parameters
    ----------
    output_dir1 : Path
        First directory containing EMOD3D velocity model files
    output_dir2 : Path
        Second directory containing EMOD3D velocity model files
    nzcvm_path : Path
        Path to the nzcvm.cfg file that defines the model dimensions
    threshold : float
        Threshold for considering differences significant (default: 1e-5)

    """

    print(f"Comparing EMOD3D output from {output_dir1} and {output_dir2}")

    # Parse the config file
    vm_params = parse_nzcvm_config(nzcvm_path)

    nx = vm_params["nx"]
    ny = vm_params["ny"]
    nz = vm_params["nz"]

    print(f"Model dimensions: nx={nx}, ny={ny}, nz={nz}")

    # Run comparison
    comparison_results = compare_output_files(
        output_dir1, output_dir2, nx, ny, nz, threshold
    )

    # Print results
    for key in comparison_results:
        print()  # for better readability
        print(f"Results for {key}:")
        print(comparison_results[key])

    # Provide a summary of critical results
    print("Summary:")
    for key in ["vp", "vs", "rho"]:
        if key in comparison_results:
            result = comparison_results[key]
            if isinstance(result, dict) and "allclose" in result:
                print(
                    f"  {key}: {'MATCH' if result['allclose'] else 'MISMATCH'} (max diff: {result.get('max_diff')})"
                )
            else:
                print(f"  {key}: Error in comparison")


if __name__ == "__main__":
    app()
