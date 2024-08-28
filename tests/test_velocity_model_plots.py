import hashlib
import subprocess
import tempfile
from pathlib import Path
from urllib import request

from velocity_modelling.scripts import plot_velocity_model


def md5sum(file: Path) -> str:
    with open(file, "rb") as file_handle:
        return hashlib.file_digest(file_handle, "md5").hexdigest()


PLOT_IMAGE_DIRECTORY = Path("wiki/images")
INPUT_FFP = Path(__file__).parent / "input"

SWEDGE_RHO3D_FILE_LINK = "https://www.dropbox.com/scl/fi/y8qf8zjq4zf9yeejsopya/rho3dfile.d?rlkey=87xt4kpac94dwf9hfo5pjepyb&st=n0x5qvit&dl=0"


def test_plot_velocity_model():
    """Check that the plot-velocity-model script produces the same output as the wiki still."""
    with tempfile.NamedTemporaryFile(suffix=".png") as output_path:
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


def test_plot_velocity_model_density():
    """Check that velocity model contents are plotted correctly."""
    with (
        tempfile.NamedTemporaryFile(suffix=".png") as output_path,
        tempfile.TemporaryDirectory() as velocity_model_directory,
    ):
        # Download the swedge rho3dfile for testing
        subprocess.check_call(
            [
                "wget",
                SWEDGE_RHO3D_FILE_LINK,
                "-O",
                str(Path(velocity_model_directory) / "rho3dfile.d"),
            ]
        )

        plot_velocity_model.plot_velocity_model(
            output_path.name,
            title="Swedge1: Density Plot",
            vm_params_ffp=INPUT_FFP / "Swedge1.yaml",
            velocity_model_ffp=Path(velocity_model_directory),
            component=plot_velocity_model.VelocityModelComponent.density,
        )
        assert md5sum(PLOT_IMAGE_DIRECTORY / "swedge1.png") == md5sum(output_path.name)
