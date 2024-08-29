import subprocess
import tempfile
from pathlib import Path

import diffimg
import pytest
import typer

from velocity_modelling.scripts import plot_velocity_model

PLOT_IMAGE_DIRECTORY = Path("wiki/images")
INPUT_FFP = Path(__file__).parent / "input"

# Kept at: QuakeCoRE/Jake Faulkner/rho3dfile in Dropbox.
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
        assert diffimg.diff(PLOT_IMAGE_DIRECTORY / "kelly.png", output_path.name) < 0.05

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
        assert (
            diffimg.diff(PLOT_IMAGE_DIRECTORY / "custom.png", output_path.name) < 0.05
        )


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
        assert (
            diffimg.diff(PLOT_IMAGE_DIRECTORY / "swedge1.png", output_path.name) < 0.05
        )

        # As above but with custom parameter settings
        plot_velocity_model.plot_velocity_model(
            output_path.name,
            extent_x=333.6,
            extent_y=229.20000000000002,
            centre_lat=-45.64459685522873,
            centre_lon=167.7643481857169,
            bearing=38.5,
            nz=250,
            resolution=0.2,
            title="Swedge1: Density Plot",
            velocity_model_ffp=Path(velocity_model_directory),
            component=plot_velocity_model.VelocityModelComponent.density,
        )
        assert (
            diffimg.diff(PLOT_IMAGE_DIRECTORY / "swedge1.png", output_path.name) < 0.05
        )


def test_failing_plot_examples():
    # Providing no input dimensions should raise an exception
    with pytest.raises(typer.Exit):
        plot_velocity_model.plot_velocity_model("bad.png")

    # Providing a velocity model without a resolution should raise an exception
    with pytest.raises(typer.Exit):
        plot_velocity_model.plot_velocity_model(
            "bad.png",
            extent_x=333.6,
            extent_y=229.20000000000002,
            centre_lat=-45.64459685522873,
            centre_lon=167.7643481857169,
            bearing=38.5,
            velocity_model_ffp=Path("bad_place"),
        )
