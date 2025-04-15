from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from velocity_modelling import velocity1d


@pytest.fixture
def sample_velocity_data() -> pd.DataFrame:
    """Create a sample velocity model DataFrame for testing."""
    return pd.DataFrame(
        {
            "width": [100.0, 200.0, 300.0],
            "Vp": [2000.0, 3000.0, 4000.0],
            "Vs": [1200.0, 1800.0, 2400.0],
            "rho": [2.2, 2.4, 2.6],
            "Qp": [100.0, 150.0, 200.0],
            "Qs": [50.0, 75.0, 100.0],
        }
    )


@pytest.fixture
def temp_parquet_file(tmp_path: Path) -> Path:
    """Create a temporary parquet file for testing."""
    return tmp_path / "velocity_model.parquet"


@pytest.fixture
def temp_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file for testing."""
    return tmp_path / "velocity_model.1d"


def test_read_velocity_model_1d(
    sample_velocity_data: pd.DataFrame, temp_parquet_file: Path
):
    """Test reading velocity model from parquet and depth calculations."""
    sample_velocity_data.to_parquet(temp_parquet_file)
    result = velocity1d.read_velocity_model_1d(temp_parquet_file)

    assert set(result.columns.tolist()) == {
        "width",
        "Vp",
        "Vs",
        "rho",
        "Qp",
        "Qs",
        "top_depth",
        "bottom_depth",
    }

    expected_top_depths = [0.0, 100.0, 300.0]
    expected_bottom_depths = [100.0, 300.0, 600.0]

    np.testing.assert_array_almost_equal(result["top_depth"], expected_top_depths)
    np.testing.assert_array_almost_equal(result["bottom_depth"], expected_bottom_depths)


def test_read_velocity_model_1d_missing_columns(temp_parquet_file: Path):
    """Test reading velocity model with missing required columns."""
    invalid_data = pd.DataFrame({"width": [100.0], "Vp": [2000.0]})
    invalid_data.to_parquet(temp_parquet_file)

    with pytest.raises(ValueError, match="Missing required columns"):
        velocity1d.read_velocity_model_1d(temp_parquet_file)


def test_read_velocity_model_1d_negative_values(temp_parquet_file: Path):
    """Test reading velocity model with negative values."""
    invalid_data = pd.DataFrame(
        {
            "width": [100.0],
            "Vp": [-2000.0],
            "Vs": [1200.0],
            "rho": [2.2],
            "Qp": [100.0],
            "Qs": [50.0],
        }
    )
    invalid_data.to_parquet(temp_parquet_file)

    with pytest.raises(ValueError, match="may not contain negative numbers"):
        velocity1d.read_velocity_model_1d(temp_parquet_file)


def test_read_velocity_model_1d_plain_text(
    sample_velocity_data: pd.DataFrame, temp_text_file: Path
):
    """Test reading velocity model from plain text format."""
    velocity1d.write_velocity_model_1d_plain_text(sample_velocity_data, temp_text_file)
    result = velocity1d.read_velocity_model_1d_plain_text(temp_text_file)

    assert set(result.columns.tolist()) == {
        "width",
        "Vp",
        "Vs",
        "rho",
        "Qp",
        "Qs",
        "top_depth",
        "bottom_depth",
    }
    pd.testing.assert_frame_equal(
        result[["width", "Vp", "Vs", "rho", "Qp", "Qs"]],
        sample_velocity_data,
        check_dtype=True,
    )


def test_read_velocity_model_1d_plain_text_invalid_header(temp_text_file: Path):
    """Test reading plain text with invalid header."""
    with open(temp_text_file, "w") as f:
        f.write("invalid\n")
        f.write("100 2000 1200 2.2 100 50\n")

    with pytest.raises(ValueError, match="Invalid or missing layer count"):
        velocity1d.read_velocity_model_1d_plain_text(temp_text_file)


def test_read_velocity_model_1d_plain_text_layer_mismatch(temp_text_file: Path):
    """Test reading plain text with layer count mismatch."""
    with open(temp_text_file, "w") as f:
        f.write("2\n")  # Claim 2 layers but write 1
        f.write("100 2000 1200 2.2 100 50\n")

    with pytest.raises(ValueError, match="does not match the header"):
        velocity1d.read_velocity_model_1d_plain_text(temp_text_file)


def test_read_velocity_model_1d_plain_text_empty_data(temp_text_file: Path):
    """Test reading plain text with no data after header."""
    with open(temp_text_file, "w") as f:
        f.write("1\n")

    with pytest.raises(
        ValueError, match="Number of velocity model layers does not match the header."
    ):
        velocity1d.read_velocity_model_1d_plain_text(temp_text_file)


def test_write_velocity_model_1d_plain_text(
    sample_velocity_data: pd.DataFrame, temp_text_file: Path
):
    """Test writing velocity model to plain text format."""
    velocity1d.write_velocity_model_1d_plain_text(sample_velocity_data, temp_text_file)

    with open(temp_text_file, "r") as f:
        lines = f.readlines()

    assert int(lines[0].strip()) == len(sample_velocity_data)

    for i, line in enumerate(lines[1:]):
        values = [float(x) for x in line.strip().split()]
        np.testing.assert_array_almost_equal(
            values,
            sample_velocity_data.iloc[i][
                ["width", "Vp", "Vs", "rho", "Qp", "Qs"]
            ].values,
        )
