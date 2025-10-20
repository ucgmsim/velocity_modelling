# tests/test_gen_thresholds.py
import csv
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from tests.conftest import env_path
from velocity_modelling.constants import TopoTypes
from velocity_modelling.scripts.generate_thresholds import generate_thresholds

# Define the scenarios
SCENARIOS = [
    "zvalues_test",
]


class ThresholdScenarioDict(TypedDict):
    """
    TypedDict for a threshold scenario dictionary

    Attributes
    ----------
    name : str
        Name of the scenario
    station_file : Path
        Path to the input station file
    benchmark_file : Path
        Path to the benchmark file
    output_path : Path
        Path to the output directory for generated files
    data_root : Path
        Path to the root directory for model data
    model_version : str
        Version of the velocity model to use
    """

    name: str
    station_file: Path
    benchmark_file: Path
    output_path: Path
    data_root: Path
    model_version: str


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
def threshold_scenario(
    tmp_path: Path, test_paths: tuple[Path, Path], request: pytest.FixtureRequest
) -> ThresholdScenarioDict:
    """
    Fixture to provide scenario data for each test

    Parameters
    ----------
    tmp_path : Path
        Temporary path for output
    test_paths : tuple[Path, Path]
        Tuple containing benchmark directory and data root
    request : pytest.FixtureRequest
        Pytest fixture request to get the current parameter

    Returns
    -------
    ThresholdScenarioDict
        Dictionary containing scenario details

    """
    scenario_name = request.param
    scenario_path = SCENARIO_DIR / scenario_name

    # The .ll file is the input station file (legacy format: lon lat name)
    station_file = scenario_path / f"{scenario_name}.ll"

    # Benchmark output file
    benchmark_file = test_paths[0] / scenario_name / f"{scenario_name}.z"

    # Output directory
    tmp_dir = env_path("JENKINS_OUTPUT_DIR") or tmp_path
    output_path = tmp_dir / scenario_name
    data_root = test_paths[1]

    return ThresholdScenarioDict(
        name=scenario_name,
        station_file=station_file,
        benchmark_file=benchmark_file,
        output_path=output_path,
        data_root=data_root,
        model_version="2.07",  # Default model version
    )


def parse_thresholds_file(threshold_file: Path) -> dict[str, dict[str, float]]:
    """
    Parse a thresholds file (CSV format).

    Parameters
    ----------
    threshold_file : Path
        Path to the thresholds file

    Returns
    -------
    dict[str, dict[str, float]]
        Dictionary mapping station names to their Z-values and sigma
        Format: {station_name: {'Z1.0': float, 'Z2.5': float, 'sigma': float}}
    """
    with open(threshold_file, "r") as f:
        reader = csv.DictReader(f)
        results = {
            row["Station_Name"]: {
                "Z1.0": float(row["Z1.0(km)"]),
                "Z2.5": float(row["Z2.5(km)"]),
                "sigma": float(row["sigma"]),
            }
            for row in reader
        }

    return results


def compare_thresholds_files(
    benchmark_file: Path, output_file: Path, threshold: float = 1e-5
) -> dict:
    """
    Compare threshold values from output file with the benchmark file.

    Parameters
    ----------
    benchmark_file : Path
        Path to the benchmark thresholds file
    output_file : Path
        Path to the output thresholds file to compare
    threshold : float, optional
        Threshold for comparison, by default 1e-5

    Returns
    -------
    dict
        Dictionary containing comparison results

    """
    benchmark_data = parse_thresholds_file(benchmark_file)
    output_data = parse_thresholds_file(output_file)

    # Check that both files have the same stations
    benchmark_stations = set(benchmark_data.keys())
    output_stations = set(output_data.keys())

    missing_stations = benchmark_stations - output_stations
    extra_stations = output_stations - benchmark_stations

    if missing_stations or extra_stations:
        return {
            "size_check": False,
            "missing_stations": list(missing_stations),
            "extra_stations": list(extra_stations),
        }

    # Collect all values for comparison
    benchmark_z1 = []
    benchmark_z2_5 = []
    benchmark_sigma = []
    output_z1 = []
    output_z2_5 = []
    output_sigma = []

    for station in sorted(benchmark_stations):
        benchmark_z1.append(benchmark_data[station]["Z1.0"])
        benchmark_z2_5.append(benchmark_data[station]["Z2.5"])
        benchmark_sigma.append(benchmark_data[station]["sigma"])

        output_z1.append(output_data[station]["Z1.0"])
        output_z2_5.append(output_data[station]["Z2.5"])
        output_sigma.append(output_data[station]["sigma"])

    # Convert to numpy arrays
    benchmark_z1 = np.array(benchmark_z1)
    benchmark_z2_5 = np.array(benchmark_z2_5)
    benchmark_sigma = np.array(benchmark_sigma)
    output_z1 = np.array(output_z1)
    output_z2_5 = np.array(output_z2_5)
    output_sigma = np.array(output_sigma)

    # Compare each field
    z1_close = np.allclose(benchmark_z1, output_z1, atol=threshold, rtol=0)
    z2_5_close = np.allclose(benchmark_z2_5, output_z2_5, atol=threshold, rtol=0)
    sigma_close = np.allclose(benchmark_sigma, output_sigma, atol=threshold, rtol=0)

    # Calculate differences
    z1_diff = np.abs(benchmark_z1 - output_z1)
    z2_5_diff = np.abs(benchmark_z2_5 - output_z2_5)
    sigma_diff = np.abs(benchmark_sigma - output_sigma)

    return {
        "size_check": True,
        "num_stations": len(benchmark_stations),
        "Z1.0": {
            "allclose": z1_close,
            "max_diff": float(np.max(z1_diff)),
            "mean_diff": float(np.mean(z1_diff)),
            "std_diff": float(np.std(z1_diff)),
        },
        "Z2.5": {
            "allclose": z2_5_close,
            "max_diff": float(np.max(z2_5_diff)),
            "mean_diff": float(np.mean(z2_5_diff)),
            "std_diff": float(np.std(z2_5_diff)),
        },
        "sigma": {
            "allclose": sigma_close,
            "max_diff": float(np.max(sigma_diff)),
            "mean_diff": float(np.mean(sigma_diff)),
            "std_diff": float(np.std(sigma_diff)),
        },
        "all_close": z1_close and z2_5_close and sigma_close,
    }


def test_gen_threshold_points(threshold_scenario: ThresholdScenarioDict):
    """
    Test generate_thresholds.py with different scenarios
    and compare outputs with benchmarks.

    Parameters
    ----------
    threshold_scenario : ThresholdScenarioDict
        Dictionary containing scenario details

    Returns
    -------
    None

    """
    # Create output directory for this scenario
    threshold_scenario["output_path"].mkdir(exist_ok=True, parents=True)

    # Expected output file name (based on script behavior with _thresholds suffix)
    expected_output_file = (
        threshold_scenario["output_path"]
        / f"{threshold_scenario['name']}_thresholds.csv"
    )

    # Call the function directly instead of subprocess
    # Use legacy station file format options for backward compatibility
    generate_thresholds(
        locations_csv=threshold_scenario["station_file"],
        lon_index=0,
        lat_index=1,
        name_index=2,  # For legacy station file format: lon lat name
        sep=" ",  # For legacy format: space-separated
        skip_rows=0,
        model_version=threshold_scenario["model_version"],
        threshold_type=None,  # Will default to [Z1.0, Z2.5]
        out_dir=threshold_scenario["output_path"],
        topo_type=TopoTypes.SQUASHED.name,
        write_no_header=False,
        nzcvm_registry=None,
        nzcvm_data_root=threshold_scenario["data_root"],
        log_level="INFO",
    )

    # Check that the output file was created
    assert expected_output_file.exists(), (
        f"Output file not created: {expected_output_file}\n"
        f"Available files: {list(threshold_scenario['output_path'].iterdir())}"
    )

    # Compare output file with benchmark
    comparison_results = compare_thresholds_files(
        threshold_scenario["benchmark_file"],
        expected_output_file,
        threshold=1e-5,
    )

    # Check size/structure
    assert comparison_results["size_check"], (
        f"Size check failed for {threshold_scenario['name']}:\n"
        f"Missing stations: {comparison_results.get('missing_stations', [])}\n"
        f"Extra stations: {comparison_results.get('extra_stations', [])}"
    )

    # Check each field
    for key in ["Z1.0", "Z2.5", "sigma"]:
        assert key in comparison_results, (
            f"Missing {key} in comparison results for {threshold_scenario['name']}"
        )
        assert comparison_results[key]["allclose"], (
            f"{key} values not close enough for {threshold_scenario['name']}:\n"
            f"Max diff: {comparison_results[key]['max_diff']}\n"
            f"Mean diff: {comparison_results[key]['mean_diff']}\n"
            f"Std diff: {comparison_results[key]['std_diff']}"
        )

    # Overall check
    assert comparison_results["all_close"], (
        f"Overall comparison failed for {threshold_scenario['name']}"
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
