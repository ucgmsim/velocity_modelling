"""View a velocity model in 3-dimensions."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import pyvista as pv
import typer

app = typer.Typer()


class Slices(StrEnum):
    "Slicing control options for the velocity model."

    XY = "xy"
    XZ = "xz"
    YZ = "yz"


@app.command(help="View a velocity model in 3D.")
def view_velocity_model(
    velocity_model_ffp: Annotated[
        Path, typer.Argument(help="The velocity model file to plot.")
    ],
    nx: Annotated[int, typer.Argument(help="The number of x-slices")],
    ny: Annotated[int, typer.Argument(help="The number of y-slices")],
    nz: Annotated[int, typer.Argument(help="The number of z-slices")],
    slices: Annotated[
        Slices, typer.Option(help="Add slice controls for the given axes.")
    ] = Slices.XY,
) -> None:
    """View a velocity model in 3 dimensions.

    Parameters
    ----------
    velocity_model_ffp : Path
        Path to the velocity model file to view.
    nx : int
        The number of x-slices.
    ny : int
        The number of y-slices.
    nz : int
        The number of z-slices.
    slices : Slices
        The slicing controls to add, defaults to x and y axis slicing.
    """
    data_vs = np.fromfile(velocity_model_ffp, "<f4")

    # reshape the data
    data_vs = data_vs.reshape([ny, nz, nx])
    data_vs = np.swapaxes(data_vs, 0, 2)
    data_vs = np.swapaxes(data_vs, 1, 2)
    data_vs = np.flip(data_vs, 2)  # reverse z axis

    vol = pv.ImageData()
    vol.dimensions = tuple(np.array(data_vs.shape) + 1)
    vol.cell_data["values"] = data_vs.flatten(order="F")

    p = pv.Plotter()

    p.add_mesh_clip_plane(vol, assign_to_axis=slices[0])
    p.add_mesh_clip_plane(vol, assign_to_axis=slices[1], invert=(slices[1] == "z"))

    p.show()
