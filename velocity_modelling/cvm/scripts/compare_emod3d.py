"""
Compare the output files from two directories containing the output files from emod3d.
Usage:
    python compare_emod3d.py output_dir1 output_dir2 vm_params

"""

import argparse
from pathlib import Path
import yaml
import numpy as np

from velocity_modelling.cvm.scripts.emod3d_2_csv import read_output_files


def compare_output_files(
    dir1: Path, dir2: Path, nx: int, ny: int, nz: int, threshold=1e-5
):
    data1 = read_output_files(dir1)
    data2 = read_output_files(dir2)

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
    parser.add_argument("vm_params", type=Path, help="Path to the vm_params.yaml file")
    args = parser.parse_args()

    output_dir1 = args.output_dir1
    output_dir2 = args.output_dir2
    vm_params_path = args.vm_params

    with open(vm_params_path, "r") as f:
        vm_params = yaml.safe_load(f)

    nx = vm_params["nx"]
    ny = vm_params["ny"]
    nz = vm_params["nz"]

    comparison_results = compare_output_files(output_dir1, output_dir2, nx, ny, nz)
    for key in comparison_results:
        print(f"Results for {key}:")
        print(comparison_results[key])
