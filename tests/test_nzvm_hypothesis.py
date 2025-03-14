import pytest
import shutil
import subprocess
from pathlib import Path
import os
import tempfile
from hypothesis import given, settings, strategies as st, assume, Verbosity
from velocity_modelling.cvm.scripts.compare_emod3d import (
    compare_output_files,
    parse_nzvm_config,
)

# Define paths
BASE_DIR = Path(__file__).parent.parent  # project root directory
SCRIPT_DIR = BASE_DIR / "velocity_modelling/cvm/scripts"


@pytest.fixture
def nzvm_c_binary_path(request) -> Path:
    """
    Get the path to the NZVM C binary from the command-line option or environment variable.
    """
    nzvm_path = request.config.getoption("--nzvm-binary-path")
    if nzvm_path is None:
        raise ValueError("NZVM binary path not provided. Use --nzvm-binary-path or NZVM_BINARY_PATH environment variable.")
    new_nzvm_path = Path(nzvm_path).resolve()
    if not new_nzvm_path.exists():
        raise ValueError(f"Provided NZVM binary path does not exist: {new_nzvm_path}")
    if not new_nzvm_path.is_file():
        raise ValueError(f"Provided NZVM binary path is not a file: {new_nzvm_path}")
    return new_nzvm_path


@pytest.fixture
def data_root_path(request) -> Path:
    """Configure DATA_ROOT based on command-line option."""
    data_root = request.config.getoption("--data-root")

    # data_root is not None
    new_data_root = Path(data_root).resolve()
    if not new_data_root.exists():
        raise ValueError(f"Provided DATA_ROOT does not exist: {new_data_root}")
    if not new_data_root.is_dir():
        raise ValueError(f"Provided DATA_ROOT is not a directory: {new_data_root}")
    data_root = new_data_root

    return data_root

# Define Hypothesis strategy for nzvm.cfg parameters
nzvm_config_strategy = st.fixed_dictionaries({
    "MODEL_VERSION": st.sampled_from(["2.03", "2.07"]),
    "EXTENT_X": st.floats(min_value=1, max_value=3),
    "EXTENT_Y": st.floats(min_value=1, max_value=3),
    "EXTENT_ZMAX": st.floats(min_value=1, max_value=10),
    "EXTENT_ZMIN": st.just(0.0),
    "EXTENT_Z_SPACING": st.floats(min_value=0.4, max_value=1.0),
    "EXTENT_LATLON_SPACING": st.floats(min_value=0.4, max_value=2.0),
    "MIN_VS": st.floats(min_value=0.1, max_value=0.5),
    "TOPO_TYPE": st.sampled_from(["BULLDOZED", "SQUASHED", "SQUASHED_TAPERED", "TRUE"]),
    "ORIGIN_LAT": st.floats(min_value=-45.0, max_value=-35.0),
    "ORIGIN_LON": st.floats(min_value=165.0, max_value=180.0),
    "ORIGIN_ROT": st.floats(min_value=-45.0, max_value=45.0),
})

def generate_nzvm_config(tmp_dir: Path, c_output_dir: Path, config_dict: dict) -> Path:
    """Generate an nzvm.cfg file from a given config dictionary"""
    config_content = f"""
CALL_TYPE=GENERATE_VELOCITY_MOD
MODEL_VERSION={config_dict['MODEL_VERSION']}
ORIGIN_LAT={config_dict['ORIGIN_LAT']}
ORIGIN_LON={config_dict['ORIGIN_LON']}
ORIGIN_ROT={config_dict['ORIGIN_ROT']}
EXTENT_X={config_dict['EXTENT_X']}
EXTENT_Y={config_dict['EXTENT_Y']}
EXTENT_ZMAX={config_dict['EXTENT_ZMAX']}
EXTENT_ZMIN={config_dict['EXTENT_ZMIN']}
EXTENT_Z_SPACING={config_dict['EXTENT_Z_SPACING']}
EXTENT_LATLON_SPACING={config_dict['EXTENT_LATLON_SPACING']}
MIN_VS={config_dict['MIN_VS']}
TOPO_TYPE={config_dict['TOPO_TYPE']}
OUTPUT_DIR={c_output_dir}
"""

    config_file = tmp_dir / "nzvm.cfg"
    with open(config_file, "w") as f:
        f.write(config_content.strip())
        f.write("\n")

    assert config_file.exists(), f"Config file {config_file} was not created"
    assert os.access(config_file, os.R_OK), f"Config file {config_file} is not readable"
    return config_file

@given(nzvm_config_strategy)
@settings(max_examples=5, deadline=None,  verbosity=Verbosity.verbose)
def test_nzvm_c_vs_python(nzvm_c_binary_path: Path, data_root_path: Path, config_dict : dict):
    """Test C binary vs Python script with Hypothesis-generated config"""
    # Filter out invalid configurations
    assume(config_dict["EXTENT_ZMAX"] > config_dict["EXTENT_ZMIN"])

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # Define output directories
        c_output_dir = tmp_dir / "C"
        python_output_dir = tmp_dir / "Python"

        # Ensure C output directory doesn't exist before running
        if c_output_dir.exists():
            shutil.rmtree(c_output_dir)

        # Generate config file
        config_file = generate_nzvm_config(tmp_dir, c_output_dir, config_dict)

        if nzvm_c_binary_path.parent != data_root_path.parent:
            raise ValueError("NZVM binary should be in the same directory as the data root")

        # Run C binary
        c_result = subprocess.run(
            [nzvm_c_binary_path, config_file],
            cwd=str(nzvm_c_binary_path.parent),
            capture_output=True,
            text=True,
        )
        assert (
            c_result.returncode == 0
        ), f"C binary failed: {c_result.stderr} (stdout: {c_result.stdout})"

        # Check C binary output
        c_velocity_model_dir = c_output_dir / "Velocity_Model"
        assert c_velocity_model_dir.exists() and any(
            c_velocity_model_dir.iterdir()
        ), f"No output found in {c_velocity_model_dir}"

        # Run Python script
        python_result = subprocess.run(
            [
                "python",
                str(SCRIPT_DIR / "nzvm.py"),
                "generate-velocity-model",
                str(config_file),
                "--out-dir",
                str(python_output_dir),
                "--data-root",
                str(data_root_path),
            ],
            capture_output=True,
            text=True,
        )
        assert (
            python_result.returncode == 0
        ), f"Python script failed: {python_result.stderr}"

        # Parse config for nx, ny, nz
        vm_params = parse_nzvm_config(config_file)
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
            assert comparison_results[key]["allclose"], (
                f"{key} data not close enough between C and Python:\n"
                f"Max diff: {comparison_results[key]['max_diff']}\n"
                f"Mean diff: {comparison_results[key]['mean_diff']}\n"
                f"Std diff: {comparison_results[key]['std_diff']}"
            )

if __name__ == "__main__":
    pytest.main(["-v"])