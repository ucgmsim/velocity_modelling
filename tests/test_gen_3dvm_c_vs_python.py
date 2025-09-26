import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import env_path
from velocity_modelling.tools.compare_emod3d import (
    compare_output_files,
    parse_nzcvm_config,
)

# Define paths
BASE_DIR = Path(__file__).parent.parent  # project root directory
SCRIPT_DIR = BASE_DIR / "velocity_modelling/scripts"
TEST_DIR = BASE_DIR / "tests"


def generate_random_nzcvm_config(tmp_path: Path, c_output_dir: Path) -> Path:
    """Generate a random but sensible nzcvm.cfg file with specified parameters"""
    # Sensible ranges and choices for parameters
    model_version = random.choice(["2.08"])  # Fixed to known valid versions
    extent_x = random.uniform(10, 100)
    extent_y = random.uniform(10, 100)
    extent_zmax = random.uniform(2, 50)
    extent_zmin = 0.0
    extent_z_spacing = random.uniform(0.1, 1.0)
    extent_latlon_spacing = random.uniform(0.1, 2)  # Updated range
    min_vs = random.uniform(0.1, 0.5)
    topo_type = random.choice(
        ["BULLDOZED", "SQUASHED", "SQUASHED_TAPERED", "TRUE"]
    )  # Fixed to a known valid value
    origin_lat = random.uniform(-45.0, -35.0)
    origin_lon = random.uniform(165.0, 180.0)
    origin_rot = random.uniform(-45.0, 45.0)

    # Config content with no spaces around =, reordered
    config_content = f"""
CALL_TYPE=GENERATE_VELOCITY_MOD
MODEL_VERSION={model_version}
ORIGIN_LAT={origin_lat}
ORIGIN_LON={origin_lon}
ORIGIN_ROT={origin_rot}
EXTENT_X={extent_x}
EXTENT_Y={extent_y}
EXTENT_ZMAX={extent_zmax}
EXTENT_ZMIN={extent_zmin}
EXTENT_Z_SPACING={extent_z_spacing}
EXTENT_LATLON_SPACING={extent_latlon_spacing}
MIN_VS={min_vs}
TOPO_TYPE={topo_type}
OUTPUT_DIR={c_output_dir}
"""

    # Write config in C binary's directory
    config_file = tmp_path / "nzcvm.cfg"
    with open(config_file, "w") as f:
        f.write(config_content.strip())
        f.write("\n")

    # Debug: Verify file exists and print content
    assert config_file.exists(), f"Config file {config_file} was not created"
    assert os.access(config_file, os.R_OK), f"Config file {config_file} is not readable"
    with open(config_file, "r") as f:
        print(f"Generated config file content:\n{f.read()}")

    return config_file


@pytest.mark.repeat(5)
def test_gen_3dvm_c_vs_python(
    tmp_path: Path,
    nzvm_binary_path: Path,  # provided by conftest.py
    data_root: Path,  # provided by conftest.py
):
    """Test C binary vs Python script with random config"""
    # Define output directories but don't create them yet
    tmp_dir = env_path("JENKINS_OUTPUT_DIR") or tmp_path

    c_output_dir = tmp_dir / "C"
    python_output_dir = tmp_dir / "Python"

    # Ensure C output directory doesn't exist before running C binary
    if c_output_dir.exists():
        shutil.rmtree(c_output_dir)

    # Generate random config with C output directory
    config_file = generate_random_nzcvm_config(tmp_dir, c_output_dir)

    # Run C binary from its directory with relative path
    c_result = subprocess.run(
        [nzvm_binary_path, config_file],  # Relative path since we're in C_BINARY_DIR
        cwd=str(nzvm_binary_path.parent),
        capture_output=True,
        text=True,
    )
    print(f"C binary return code: {c_result.returncode}")
    print(f"C binary stdout: {c_result.stdout}")
    print(f"C binary stderr: {c_result.stderr}")
    assert c_result.returncode == 0, (
        f"C binary failed with return code {c_result.returncode}: {c_result.stderr} (stdout: {c_result.stdout})"
    )

    # Check C binary output
    c_velocity_model_dir = c_output_dir / "Velocity_Model"
    assert c_velocity_model_dir.exists() and any(c_velocity_model_dir.iterdir()), (
        f"No output found in {c_velocity_model_dir}"
    )
    print(f"C binary wrote to expected directory: {c_velocity_model_dir}")

    print(f"=== TEST DEBUG ===")
    print(f"data_root fixture value: {data_root}")
    print(f"data_root type: {type(data_root)}")
    print(f"data_root exists: {data_root.exists()}")
    print(f"Full subprocess command: ['python',{str(SCRIPT_DIR)} / 'generate_3d_model.py', {str(config_file)}, '--out-dir',{str(python_output_dir)}, '--nzcvm-data-root', {str(data_root)}]")
    print("==================")

    # Run Python script, overriding output directory
    python_result = subprocess.run(
        [
            "python",
            str(SCRIPT_DIR / "generate_3d_model.py"),
            str(config_file),
            "--out-dir",
            str(python_output_dir),
            "--nzcvm-data-root",
            str(data_root),
        ],
        capture_output=True,
        text=True,
    )

    assert python_result.returncode == 0, (
        f"Python script failed: {python_result.stderr}"
    )

    # Parse config for nx, ny, nz (dynamically determined)
    vm_params = parse_nzcvm_config(config_file)
    nx = vm_params["nx"]
    ny = vm_params["ny"]
    nz = vm_params["nz"]

    # Compare outputs
    comparison_results = compare_output_files(
        c_velocity_model_dir, python_output_dir, nx, ny, nz, threshold=1e-5
    )

    # Check critical files (vp, vs, rho) are allclose
    for key in ["vp", "vs", "rho"]:
        assert key in comparison_results, f"Missing {key} in comparison results"
        assert comparison_results[key]["size_check"], f"Size check failed for {key}"
        assert comparison_results[key]["allclose"], (
            f"{key} data not close enough between C and Python:\n"
            f"Max diff: {comparison_results[key]['max_diff']}\n"
            f"Mean diff: {comparison_results[key]['mean_diff']}\n"
            f"Std diff: {comparison_results[key]['std_diff']}"
        )


if __name__ == "__main__":
    pytest.main(["-v"])
