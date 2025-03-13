import shutil
import subprocess
from pathlib import Path

import pytest

from velocity_modelling.cvm.scripts.compare_emod3d import (
    compare_output_files,
    parse_nzvm_config,
)

# Define the scenarios
SCENARIOS = [
    "Cant1D_2p03",
    "Cant1D_2p07",
    "Wellington_2p03",
    "Wellington_2p07",
    "Multi_Boundary",
]

# Define paths based on your directory structure
BASE_DIR = Path(__file__).parent.parent  # # project root directory
SCRIPT_DIR = BASE_DIR / "velocity_modelling/cvm/scripts"
TEST_DIR = BASE_DIR / "tests"
SCENARIO_DIR = TEST_DIR / "scenarios"
BENCHMARK_DIR = TEST_DIR / "benchmarks"




@pytest.fixture(params=SCENARIOS)
def scenario(tmp_path: Path, request):
    """Fixture to provide scenario data for each test"""
    scenario_name = request.param
    scenario_path = SCENARIO_DIR / scenario_name
    config_file = scenario_path / "nzvm.cfg"
    benchmark_path = BENCHMARK_DIR / scenario_name
    output_path = tmp_path / scenario_name

    return {
        "name": scenario_name,
        "config_file": config_file,
        "benchmark_path": benchmark_path,
        "output_path": output_path,
    }


def test_nzvm_scenarios(scenario):
    """
    Test generate_velocity_model.py with different scenarios
    and compare outputs with benchmarks
    """
    # Create output directory for this scenario
    scenario["output_path"].mkdir(exist_ok=True)

    # Run the generate_velocity_model.py script with --out_dir
    result = subprocess.run(
        [
            "python",
            str(SCRIPT_DIR / "nzvm.py"),
            "generate-velocity-model",
            str(scenario["config_file"]),
            "--out-dir",
            str(scenario["output_path"]),
        ],
        capture_output=True,
        text=True,
    )

    # Check if the script ran successfully
    assert (
        result.returncode == 0
    ), f"Script failed for {scenario['name']}: {result.stderr}"

    # Parse the config file to get nx, ny, nz
    vm_params = parse_nzvm_config(scenario["config_file"])
    nx = vm_params["nx"]
    ny = vm_params["ny"]
    nz = vm_params["nz"]

    # Compare output files with benchmarks
    comparison_results = compare_output_files(
        scenario["benchmark_path"],
        scenario["output_path"],
        nx,
        ny,
        nz,
        threshold=1e-5,  # Using the default threshold from compare_emod3d.py
    )

    # Check critical files (vp, vs, rho) are allclose
    for key in ["vp", "vs", "rho"]:
        assert (
            key in comparison_results
        ), f"Missing {key} in comparison results for {scenario['name']}"
        assert comparison_results[key]["allclose"], (
            f"{key} data not close enough for {scenario['name']}:\n"
            f"Max diff: {comparison_results[key]['max_diff']}\n"
            f"Mean diff: {comparison_results[key]['mean_diff']}\n"
            f"Std diff: {comparison_results[key]['std_diff']}"
        )

    # Note: We ignore inbasin comparison as the original C code saves incorrect values


if __name__ == "__main__":
    pytest.main(["-v"])
