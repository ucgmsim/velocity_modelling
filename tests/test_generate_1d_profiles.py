from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from velocity_modelling.scripts.generate_1d_profiles import generate_1d_profiles
from tests.conftest import env_path

# Define paths relative to the test file if needed, but fixtures are better
BASE_DIR = Path(__file__).parent
SCENARIO_DIR = BASE_DIR / "scenarios" / "1d_profiles"
BENCHMARK_SUBDIR = "1d_profiles"


def compare_profiles(benchmark_file: Path, output_file: Path):
    """
    Compare two profile files.
    """
    # Read files, skipping the first 5 lines of metadata
    # The separate is tab, engine='python' might be safer for varying whitespace but \t seems explicit.
    # We strip whitespace from headers usually.

    # Looking at the file content:
    # Elevation (km) \t Vp (km/s) \t Vs (km/s) \t Rho (g/cm^3)
    # The columns might have extra spaces.

    df_bench = pd.read_csv(benchmark_file, sep=r"\s+", skiprows=4, header=0)
    df_out = pd.read_csv(output_file, sep=r"\s+", skiprows=4, header=0)

    # We use sep=r"\s+" because the file might be visually aligned with spaces/tabs.
    # The header line is line 5 (0-indexed index 4).
    # wait, line 1 is "Properties...", line 2 "Model...", line 3 "Topo...", line 4 "Minimum...", line 5 "Elevation..."
    # So skiprows should be 4 if we want line 5 to be header?
    # Let's verify:
    # 0: Prop
    # 1: Model
    # 2: Topo
    # 3: Min
    # 4: Elevation (Header)

    # However, read_csv skiprows argument: "Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file."
    # If skiprows=4, it skips lines 0,1,2,3. Line 4 is the first line read (header).

    # Also need to check if columns match.
    pd.testing.assert_frame_equal(df_bench, df_out, check_dtype=False)


def test_generate_1d_profiles(tmp_path: Path, benchmark_dir: Path, data_root: Path):
    """
    Test generate_1d_profiles against benchmarks.
    """
    scenario_dir = SCENARIO_DIR
    locations_csv = scenario_dir / "locations.csv"

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Run the generation
    generate_1d_profiles(
        location_csv=locations_csv,
        out_dir=output_dir,
        model_version="2.09",
        topo_type="TRUE",
        nzcvm_data_root=data_root,
        min_vs=0.0,
    )

    # Read locations to know what files to expect
    df_loc = pd.read_csv(locations_csv, skipinitialspace=True)
    df_loc.columns = df_loc.columns.str.lower()

    profiles_dir = output_dir / "Profiles"
    assert profiles_dir.exists()

    benchmark_profiles_dir = benchmark_dir / BENCHMARK_SUBDIR

    for _, row in df_loc.iterrows():
        profile_id = row["id"]
        filename = f"Profile_{profile_id}.txt"

        output_file = profiles_dir / filename
        benchmark_file = benchmark_profiles_dir / filename

        assert output_file.exists(), f"Output file {filename} not found"
        assert benchmark_file.exists(), f"Benchmark file {filename} not found"

        print(f"Comparing {filename}...")
        compare_profiles(benchmark_file, output_file)
