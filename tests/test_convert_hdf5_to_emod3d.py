# tests/test_convert_hdf5_to_emod3d.py
"""Unit tests for convert_hdf5_to_emod3d."""
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from velocity_modelling.tools.convert_hdf5_to_emod3d import convert_hdf5_to_emod3d


def _make_hdf5(path: Path, nz: int, ny: int, nx: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Create a minimal HDF5 velocity-model file and return the arrays written."""
    arrays = {
        "/properties/vp": rng.random((nz, ny, nx), dtype=np.float64).astype(np.float32),
        "/properties/vs": rng.random((nz, ny, nx), dtype=np.float64).astype(np.float32),
        "/properties/rho": rng.random((nz, ny, nx), dtype=np.float64).astype(np.float32),
        "/properties/inbasin": rng.random((nz, ny, nx), dtype=np.float64).astype(np.float32),
    }
    with h5py.File(path, "w") as hf:
        for dset_path, data in arrays.items():
            hf.create_dataset(dset_path, data=data)
    return arrays


def _read_binary(path: Path, nz: int, ny: int, nx: int) -> np.ndarray:
    """Read a flat binary float32 file back into a (nz, ny, nx) array."""
    raw = np.frombuffer(path.read_bytes(), dtype=np.float32)
    # File layout: ny slices of shape (nz, nx), concatenated
    return raw.reshape(ny, nz, nx).transpose(1, 0, 2)  # -> (nz, ny, nx)


class TestConvertHdf5ToEmod3d:
    def test_output_matches_input(self, tmp_path: Path) -> None:
        """Binary files must reproduce the HDF5 data exactly."""
        nz, ny, nx = 3, 4, 5
        rng = np.random.default_rng(42)
        src = tmp_path / "model.h5"
        arrays = _make_hdf5(src, nz, ny, nx, rng)

        convert_hdf5_to_emod3d(src, tmp_path)

        file_map = {
            "/properties/vp": tmp_path / "vp3dfile.p",
            "/properties/vs": tmp_path / "vs3dfile.s",
            "/properties/rho": tmp_path / "rho3dfile.d",
            "/properties/inbasin": tmp_path / "in_basin_mask.b",
        }
        for dset_path, out_file in file_map.items():
            assert out_file.exists(), f"{out_file.name} was not created"
            result = _read_binary(out_file, nz, ny, nx)
            np.testing.assert_array_equal(
                result,
                arrays[dset_path],
                err_msg=f"Data mismatch for {dset_path}",
            )

    def test_no_temp_files_after_success(self, tmp_path: Path) -> None:
        """No .tmp files should remain after a successful conversion."""
        nz, ny, nx = 2, 2, 2
        rng = np.random.default_rng(0)
        src = tmp_path / "model.h5"
        _make_hdf5(src, nz, ny, nx, rng)

        convert_hdf5_to_emod3d(src, tmp_path)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Unexpected .tmp files left behind: {tmp_files}"

    def test_cleanup_on_failure(self, tmp_path: Path) -> None:
        """Temp files must be removed and the exception re-raised on thread failure."""
        nz, ny, nx = 2, 2, 2
        rng = np.random.default_rng(1)
        src = tmp_path / "model.h5"
        _make_hdf5(src, nz, ny, nx, rng)

        boom = RuntimeError("simulated I/O error")

        original_convert = __import__(
            "velocity_modelling.tools.convert_hdf5_to_emod3d",
            fromlist=["_convert_dataset"],
        )._convert_dataset

        call_count = {"n": 0}

        def failing_convert(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise boom
            return original_convert(*args, **kwargs)

        with patch(
            "velocity_modelling.tools.convert_hdf5_to_emod3d._convert_dataset",
            side_effect=failing_convert,
        ):
            with pytest.raises(RuntimeError, match="simulated I/O error"):
                convert_hdf5_to_emod3d(src, tmp_path)

        # No .tmp files should be left
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Leftover .tmp files after failure: {tmp_files}"

        # No final output files should be present (atomic rename never happened)
        for name in ("vp3dfile.p", "vs3dfile.s", "rho3dfile.d", "in_basin_mask.b"):
            assert not (tmp_path / name).exists(), f"{name} should not exist after failure"
