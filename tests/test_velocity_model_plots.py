import hashlib
import tempfile
from pathlib import Path

from velocity_modelling.scripts import plot_velocity_model


def md5sum(file: Path) -> str:
    with open(file, "rb") as file_handle:
        return hashlib.file_digest(file_handle, "md5").hexdigest()


PLOT_IMAGE_DIRECTORY = Path("wiki/images")
INPUT_FFP = Path(__file__).parent / "input"


def test_plot_velocity_model():
    """Check that the plot-velocity-model script produces the same output as the wiki still."""
    with tempfile.NamedTemporaryFile(suffix=".png") as output_path:
        # TODO: uncomment when the new realisation format is ready.
        # plot_velocity_model.plot_velocity_model(
        #     output_path.name,
        #     realisation_ffp=INPUT_FFP / "rupture_1.json",
        #     title="Rupture 1",
        #     latitude_pad=0.5,
        #     longitude_pad=0.5,
        # )
        # assert md5sum(PLOT_IMAGE_DIRECTORY / "rupture_1.png") == md5sum(
        #     output_path.name
        # )

        plot_velocity_model.plot_velocity_model(
            output_path.name,
            vm_params_ffp=INPUT_FFP / "Kelly.yaml",
            title="Kelly",
            latitude_pad=0.5,
            longitude_pad=0.5,
        )
        assert md5sum(PLOT_IMAGE_DIRECTORY / "kelly.png") == md5sum(output_path.name)

        plot_velocity_model.plot_velocity_model(
            output_path.name,
            centre_lat=-43,
            centre_lon=172,
            extent_x=100,
            extent_y=100,
            bearing=45,
            title="Sample velocity model",
            latitude_pad=0.5,
            longitude_pad=0.5,
        )
        assert md5sum(PLOT_IMAGE_DIRECTORY / "custom.png") == md5sum(output_path.name)
