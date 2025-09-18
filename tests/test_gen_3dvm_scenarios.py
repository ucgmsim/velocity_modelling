# tests/test_gen_3dvm_scenarios.py
import os
import subprocess
from pathlib import Path
from typing import TypedDict

import pytest

from velocity_modelling.tools.compare_emod3d import (
    compare_output_files,
    parse_nzcvm_config,
)

# Define the scenarios
SCENARIOS = [
    "Cant1D_2p03",
    "Cant1D_2p07",
    "Wellington_2p03",
    "Wellington_2p07",
    "Multi_Boundary",
]


class ScenarioDict(TypedDict):
    """
    TypedDict for a scenario dictionary
    """

    name: str
    config_file: Path
    benchmark_path: Path
    output_path: Path
    data_root: Path


# Define paths based on your directory structure
BASE_DIR = Path(__file__).parent.parent  # Project root directory
SCRIPT_DIR = BASE_DIR / "velocity_modelling/scripts"
TEST_DIR = BASE_DIR / "tests"
SCENARIO_DIR = TEST_DIR / "scenarios"
BENCHMARK_DIR = TEST_DIR / "benchmarks"  # Default value, can be overridden


@pytest.fixture
def test_paths(benchmark_dir: Path, data_root: Path) -> tuple[Path, Path]:
    return benchmark_dir, data_root


@pytest.fixture(params=SCENARIOS)
def scenario(
    tmp_path: Path, test_paths: tuple[Path, Path], request: pytest.FixtureRequest
) -> ScenarioDict:
    """Fixture to provide scenario data for each test"""
    scenario_name = request.param
    scenario_path = SCENARIO_DIR / scenario_name
    config_file = scenario_path / "nzcvm.cfg"
    benchmark_path = test_paths[0] / scenario_name
    output_path = Path(os.environ.get("JENKINS_OUTPUT_DIR", tmp_path)) / scenario_name
    data_root = test_paths[1]

    return ScenarioDict(
        name=scenario_name,
        config_file=config_file,
        benchmark_path=benchmark_path,
        output_path=output_path,
        data_root=data_root,
    )


def test_gen_3dvm_scenarios(scenario: ScenarioDict):
    """
    Test generate_3d_model.py with different scenarios
    and compare outputs with benchmarks
    """

    # Create output directory for this scenario
    scenario["output_path"].mkdir(exist_ok=True)

    # Run the generate_3d_model.py script with --out-dir
    result = subprocess.run(
        [
            "python",
            str(SCRIPT_DIR / "generate_3d_model.py"),
            str(scenario["config_file"]),
            "--out-dir",
            str(scenario["output_path"]),
            "--nzcvm-data-root",
            str(scenario["data_root"]),
        ],
        capture_output=True,
        text=True,
    )

    # Check if the script ran successfully
    assert result.returncode == 0, (
        f"Script failed for {scenario['name']}: {result.stderr}"
    )

    # Parse the config file to get nx, ny, nz
    vm_params = parse_nzcvm_config(scenario["config_file"])
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
        assert key in comparison_results, (
            f"Missing {key} in comparison results for {scenario['name']}"
        )
        assert comparison_results[key]["size_check"], (
            f"Size check failed for {key} in {scenario['name']}"
        )
        assert comparison_results[key]["allclose"], (
            f"{key} data not close enough for {scenario['name']}:\n"
            f"Max diff: {comparison_results[key]['max_diff']}\n"
            f"Mean diff: {comparison_results[key]['mean_diff']}\n"
            f"Std diff: {comparison_results[key]['std_diff']}"
        )
        # Note: We ignore inbasin comparison as the original C code saves incorrect values


if __name__ == "__main__":
    pytest.main(["-v"])
