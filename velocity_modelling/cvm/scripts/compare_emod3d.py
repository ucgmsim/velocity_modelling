"""
Compare the output files from two directories containing the output files from emod3d.

This is still experimental

Usage:
    python compare_emod3d.py output_dir1 output_dir2 nzvm.cfg

Example output

Comparing vp with length 2250000 Data1 len=2250000 Data2 len=2250000
Data1: max=8.906652450561523, min=1.7999999523162842 mean=6.716171741485596 std=1.10097074508667
Data2: max=8.906652450561523, min=1.7999999523162842 mean=6.716171741485596 std=1.10097074508667
Comparing vs with length 2250000 Data1 len=2250000 Data2 len=2250000
Data1: max=5.042928695678711, min=0.5 mean=3.8954577445983887 std=0.5903079509735107
Data2: max=5.042928695678711, min=0.5 mean=3.8954577445983887 std=0.5903079509735107
Comparing rho with length 2250000 Data1 len=2250000 Data2 len=2250000
Data1: max=3.567366361618042, min=1.809999942779541 mean=2.980178117752075 std=0.2874918580055237
Data2: max=3.567366361618042, min=1.809999942779541 mean=2.980178117752075 std=0.2874918580055237
Comparing inbasin with length 2250000 Data1 len=2250000 Data2 len=2250000
Data1: max=27, min=-1 mean=-0.9775875555555555 std=0.37502816798004596
Data2: max=41, min=0 mean=0.013234222222222222 std=0.3260367563224791
Results for vp:
{'allclose': True, 'difference': array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), 'max_diff': 9.536743e-07, 'min_diff': 0.0, 'mean_diff': 3.390842e-12, 'std_diff': 1.4216468e-09, 'first_significant_index': None, 'x_index': None, 'y_index': None, 'z_index': None}
Results for vs:
{'allclose': True, 'difference': array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), 'max_diff': 4.7683716e-07, 'min_diff': 0.0, 'mean_diff': 1.695421e-12, 'std_diff': 6.743459e-10, 'first_significant_index': None, 'x_index': None, 'y_index': None, 'z_index': None}
Results for rho:
{'allclose': True, 'difference': array([0., 0., 0., ..., 0., 0., 0.], dtype=float32), 'max_diff': 2.3841858e-07, 'min_diff': 0.0, 'mean_diff': 1.4834933e-12, 'std_diff': 5.947172e-10, 'first_significant_index': None, 'x_index': None, 'y_index': None, 'z_index': None}
Results for inbasin:
{'allclose': False, 'difference': array([1, 1, 1, ..., 1, 1, 1], dtype=int8), 'max_diff': 14, 'min_diff': 0, 'mean_diff': 0.9908217777777778, 'std_diff': 0.0979692934498443, 'first_significant_index': 0, 'x_index': 0, 'y_index': 0, 'z_index': 0}


Just focus on 'allclose' for vp, vs and rho.  inbasin from the original C code is incorrect, so it will always be False.

"""

import argparse
from pathlib import Path

import numpy as np

from velocity_modelling.cvm.scripts.generate_velocity_model import (
    parse_nzvm_config,
)
from velocity_modelling.cvm.write.emod3d import (
    read_emomd3d_vm,
)


def compare_output_files(
    dir1: Path, dir2: Path, nx: int, ny: int, nz: int, threshold: float = 1e-5
):
    """
    Compare the output files from two directories containing the output files from emod3d.

    Parameters
    ----------
    dir1: Path
    dir2: Path
    nx: int
    ny: int
    nz: int
    threshold: float, optional
        The threshold for the difference between the two files. The default is 1e-5.

    Returns
    -------
    dict
    """
    data1 = read_emomd3d_vm(dir1)
    data2 = read_emomd3d_vm(dir2)

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
            }

        else:
            comparison[key] = "File missing in second directory"

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare write files from two directories."
    )
    parser.add_argument(
        "output_dir1", type=Path, help="First directory containing the write files."
    )
    parser.add_argument(
        "output_dir2", type=Path, help="Second directory containing the write files."
    )
    parser.add_argument("nzvm_path", type=Path, help="Path to the nzvm.cfg file")
    args = parser.parse_args()

    output_dir1 = args.output_dir1
    output_dir2 = args.output_dir2
    nzvm_path = args.nzvm_path

    # Parse the config file
    vm_params = parse_nzvm_config(nzvm_path)

    nx = vm_params["nx"]
    ny = vm_params["ny"]
    nz = vm_params["nz"]

    comparison_results = compare_output_files(output_dir1, output_dir2, nx, ny, nz)
    for key in comparison_results:
        print(f"Results for {key}:")
        print(comparison_results[key])
