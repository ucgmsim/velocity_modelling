import csv
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

BASE_DIR = Path(__file__).parent.parent  # project root directory
SCRIPT_DIR = BASE_DIR / "velocity_modelling/scripts"
TEST_DIR = BASE_DIR / "tests"

MODEL_VERSIONS = ["2.08"]  # Fixed to known valid versions


@pytest.fixture
def nzvm_c_binary_path(request: pytest.FixtureRequest) -> Path:
    """
    Get the path to the nzcvm C binary from the command-line option or environment variable.
    """
    nzvm_path = request.config.getoption("--nzvm-binary-path")
    if nzvm_path is None:
        raise ValueError(
            "nzvm binary path not provided. Use --nzvm-binary-path or NZVM_BINARY_PATH environment variable."
        )
    new_nzvm_path = Path(nzvm_path).resolve()
    if not new_nzvm_path.exists():
        raise ValueError(f"Provided nzvm binary path does not exist: {new_nzvm_path}")
    if not new_nzvm_path.is_file():
        raise ValueError(f"Provided nzvm binary path is not a file: {new_nzvm_path}")
    return new_nzvm_path


def generate_random_profile_config(
    tmp_path: Path, output_dir: Path
) -> tuple[Path, dict]:
    """Generate a random profile config file for the C version."""
    profile_id = random.randint(100, 999)
    profile_lat = random.uniform(-45.0, -35.0)
    profile_lon = random.uniform(165.0, 180.0)
    profile_zmin = round(random.uniform(-0.05, 0.0), 3)
    profile_zmax = round(random.uniform(0.5, 3.0), 3)
    spacing = round(random.uniform(0.001, 0.05), 5)
    min_vs = round(random.uniform(0.0, 0.5), 3)
    topo_type = random.choice(["TRUE", "BULLDOZED", "SQUASHED", "SQUASHED_TAPERED"])
    model_version = random.choice(MODEL_VERSIONS)

    output_subdir = output_dir / f"c_output/{profile_id}"
    # Ensure C output directory doesn't exist before running C binary
    if output_subdir.exists():
        shutil.rmtree(output_subdir)
    output_subdir.parent.mkdir(parents=True, exist_ok=True)

    config = f"""
CALL_TYPE=GENERATE_PROFILE
MODEL_VERSION={model_version}
OUTPUT_DIR={output_subdir}
PROFILE_LAT={profile_lat}
PROFILE_LON={profile_lon}
PROFILE_ZMIN={profile_zmin}
PROFILE_ZMAX={profile_zmax}
EXTENT_Z_SPACING_PROFILE={spacing}
PROFILE_MIN_VS={min_vs}
TOPO_TYPE={topo_type}
    """

    config_path = tmp_path / f"nzcvm_profile_{profile_id}.cfg"
    with open(config_path, "w") as f:
        f.write(config.strip())
        f.write("\n")

    params = {
        "id": profile_id,
        "lat": profile_lat,
        "lon": profile_lon,
        "zmin": profile_zmin,
        "zmax": profile_zmax,
        "spacing": spacing,
        "min_vs": min_vs,
        "topo_type": topo_type,
        "model_version": model_version,
    }
    return config_path, params


def generate_location_csv(csv_path: Path, params: dict):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "lon", "lat", "zmin", "zmax", "spacing"])
        writer.writerow(
            [
                params["id"],
                params["lon"],
                params["lat"],
                params["zmin"],
                params["zmax"],
                params["spacing"],
            ]
        )


def read_profile_data(file_path: Path) -> np.ndarray:
    with open(file_path, "r") as f:
        lines = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
    header_idx = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("Depth") or line.startswith("Elevation")
    )
    data_lines = lines[header_idx + 1 :]
    data = np.genfromtxt(data_lines)
    return data


def compare_profiles(c_path: Path, py_path: Path, atol: float = 1e-5):
    c_data = read_profile_data(c_path)
    py_data = read_profile_data(py_path)

    assert c_data.shape == py_data.shape, (
        f"Shape mismatch: {c_data.shape} vs {py_data.shape}"
    )

    # Check that NaNs are in the same positions
    c_nan_mask = np.isnan(c_data)
    py_nan_mask = np.isnan(py_data)
    assert np.array_equal(c_nan_mask, py_nan_mask), (
        "NaN positions mismatch between C and Python outputs"
    )

    # Compare only non-NaN values
    mask = ~c_nan_mask
    diff = np.abs(c_data[mask] - py_data[mask])
    assert np.all(diff < atol), (
        f"Profile mismatch. Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}"
    )


def read_surface_depths(file_path: Path) -> dict:
    surface_depths = {}
    with open(file_path, "r") as f:
        for line in f:
            line = re.sub(r"\s+", " ", line).strip()

            if not line or line.startswith(
                ("Surface", "Global", "Basin", "#")
            ):  # skip known headers
                continue
            parts = line.split()
            try:
                parts.remove(
                    "-"
                )  # Remove any leading dashes that may be present in Python output
            except ValueError:
                pass  # Ignore if no dashes are present

            if len(parts) >= 2:
                try:
                    name = parts[0]
                    value = float(parts[-1])
                    surface_depths[name] = value
                except ValueError:
                    continue
    return surface_depths


def compare_surface_depths(c_path: Path, py_path: Path, atol: float = 1e-5):
    c_dict = read_surface_depths(c_path)
    py_dict = read_surface_depths(py_path)

    matched = 0
    print(c_dict)
    print(py_dict)
    assert len(c_dict) == len(py_dict), f"Mismatch: {len(c_dict)} vs {py_dict}"
    c_items = list(c_dict.items())
    py_items = list(py_dict.items())

    for i, (c_key, c_val) in enumerate(c_items):
        py_key, py_val = py_items[i]
        assert abs(c_val - py_val) < atol, (
            f"Mismatch in surface '{c_key}': {c_val} (C) vs '{py_key}' : {py_val} (Python)"
        )
        matched += 1
        break

        assert matched > 0, "No matching surface names found between files"


@pytest.mark.repeat(5)
def test_c_vs_python_profile(
    tmp_path: Path,
    nzvm_c_binary_path: Path,
):
    root_out_dir = tmp_path / "1d_profiles"
    config_path, params = generate_random_profile_config(tmp_path, root_out_dir)

    # Run C binary from its directory with relative path
    c_result = subprocess.run(
        [
            nzvm_c_binary_path,
            str(config_path),
        ],  # Relative path since we're in C_BINARY_DIR
        cwd=str(nzvm_c_binary_path.parent),
        capture_output=True,
        text=True,
    )

    print(f"C binary return code: {c_result.returncode}")
    print(f"C binary stdout: {c_result.stdout}")
    print(f"C binary stderr: {c_result.stderr}")
    assert c_result.returncode == 0, (
        f"C binary failed with return code {c_result.returncode}: {c_result.stderr} (stdout: {c_result.stdout})"
    )

    # Run Python version
    location_csv = tmp_path / "locations.csv"
    generate_location_csv(location_csv, params)
    subprocess.run(
        [
            "python",
            str(SCRIPT_DIR / "generate_1d_profiles.py"),
            "--model-version",
            params["model_version"],
            "--out-dir",
            str(root_out_dir / "output"),
            "--location-csv",
            str(location_csv),
            "--topo-type",
            params["topo_type"],
            "--min-vs",
            str(params["min_vs"]),
        ],
        check=True,
    )

    # Compare Profile
    cid = params["id"]
    compare_profiles(
        root_out_dir / f"c_output/{cid}/Profile/Profile.txt",
        root_out_dir / f"output/Profiles/Profile_{cid}.txt",
    )
    compare_surface_depths(
        root_out_dir / f"c_output/{cid}/Profile/ProfileSurfaceDepths.txt",
        root_out_dir / f"output/Profiles/ProfileSurfaceDepths_{cid}.txt",
    )


if __name__ == "__main__":
    pytest.main(["-v"])
