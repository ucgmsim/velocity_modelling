from pathlib import Path

import pandas as pd
import pytest

from velocity_modelling.scripts.generate_1d_profiles import generate_1d_profiles

# Define paths
BASE_DIR = Path(__file__).parent
SCENARIO_DIR = BASE_DIR / "scenarios" / "1d_profiles"
BENCHMARK_SUBDIR = "1d_profiles"
LOCATIONS_CSV = SCENARIO_DIR / "locations.csv"


# Read locations at module level for parametrization
def get_profile_ids():
    if not LOCATIONS_CSV.exists():
        return []
    df = pd.read_csv(LOCATIONS_CSV, skipinitialspace=True)
    return df["id"].tolist()


PROFILE_IDS = get_profile_ids()


def compare_profiles(benchmark_file: Path, output_file: Path):
    """
    Compare two profile files.
    """
    assert output_file.exists(), f"Output file {output_file.name} not found"
    assert benchmark_file.exists(), f"Benchmark file {benchmark_file.name} not found"

    df_bench = pd.read_csv(benchmark_file, sep=r"\s+", skiprows=4, header=0)
    df_out = pd.read_csv(output_file, sep=r"\s+", skiprows=4, header=0)

    pd.testing.assert_frame_equal(df_bench, df_out, check_dtype=False)


@pytest.fixture(scope="module")
def generated_profiles_dir(
    tmp_path_factory: pytest.TempPathFactory, data_root: Path
) -> Path:
    """
    Run generate_1d_profiles once and return the output directory.

    NOTE: tmp_path_factory is a built-in pytest fixture that is injected automatically.
    We import it implicitly by naming the argument 'tmp_path_factory'.
    Since this fixture is module-scoped, we use tmp_path_factory (session-scoped)
    instead of tmp_path (function-scoped) to create a temp directory that persists
    for all tests in this module.
    """
    if not PROFILE_IDS:
        pytest.skip("No profiles found in locations.csv")

    output_dir = tmp_path_factory.mktemp("1d_profiles_output")

    # Run the generation
    generate_1d_profiles(
        location_csv=LOCATIONS_CSV,
        out_dir=output_dir,
        model_version="2.09",
        topo_type="TRUE",
        nzcvm_data_root=data_root,
        min_vs=0.0,
    )

    return output_dir


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_profile_verification(
    profile_id: str, generated_profiles_dir: Path, benchmark_dir: Path
) -> None:
    """
    Test individual profile against benchmark.

    This test function is parametrized by `profile_id`, treating each location
    in the input CSV as a separate test case. This ensures granular reporting:
    if one profile fails verification, others can still pass.
    """
    filename = f"Profile_{profile_id}.txt"

    output_file = generated_profiles_dir / "Profiles" / filename
    benchmark_file = benchmark_dir / BENCHMARK_SUBDIR / filename

    compare_profiles(benchmark_file, output_file)
