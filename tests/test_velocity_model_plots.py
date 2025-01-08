import subprocess
from pathlib import Path

import diffimg

from velocity_modelling.scripts import plot_velocity_model

PLOT_IMAGE_DIRECTORY = Path("wiki/images")
INPUT_FFP = Path(__file__).parent / "input"

# Kept at: /QuakeCoRE/Public/rho3dfile in Dropbox.
SWEDGE_RHO3D_FILE_LINK = "https://www.dropbox.com/scl/fi/y8qf8zjq4zf9yeejsopya/rho3dfile.d?rlkey=87xt4kpac94dwf9hfo5pjepyb&st=n0x5qvit&dl=0"


def test_plot_velocity_model(tmp_path: Path):
    """Check that the plot-velocity-model script produces the same output as the wiki still."""
    output_plot = tmp_path / "plot.png"
    plot_velocity_model.plot_vm_params(
        INPUT_FFP / "Kelly.yaml",
        output_plot,
        title="Kelly",
        latitude_pad=0.5,
        longitude_pad=0.5,
    )
    assert diffimg.diff(PLOT_IMAGE_DIRECTORY / "kelly.png", output_plot) < 0.05


def test_plot_velocity_model_density(tmp_path: Path):
    """Check that velocity model contents are plotted correctly."""
    velocity_model_directory = tmp_path / "velocity_model"
    velocity_model_directory.mkdir()
    output_path = tmp_path / "plot.png"
    # Download the swedge rho3dfile for testing
    subprocess.check_call(
        [
            "wget",
            SWEDGE_RHO3D_FILE_LINK,
            "-O",
            str(velocity_model_directory / "rho3dfile.d"),
        ]
    )

    plot_velocity_model.plot_vm_params(
        INPUT_FFP / "Swedge1.yaml",
        output_path,
        title="Swedge1: Density Plot",
        velocity_model_ffp=velocity_model_directory,
        component=plot_velocity_model.VelocityModelComponent.density,
    )
    assert diffimg.diff(PLOT_IMAGE_DIRECTORY / "swedge1.png", output_path) < 0.05
