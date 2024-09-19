from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import pygmt
import typer
import yaml

from pygmt_helper import plotting
from qcore import coordinates, grid
from velocity_modelling import bounding_box
from velocity_modelling.bounding_box import BoundingBox

app = typer.Typer()


class VelocityModelComponent(str, Enum):
    """Enumeration for different components of a velocity model.

    Attributes
    ----------
    p_wave : str
        P-wave (Primary wave) velocity component.
    s_wave : str
        S-wave (Secondary wave) velocity component.
    density : str
        Density component of the velocity model.
    """

    p_wave = "p_wave"
    s_wave = "s_wave"
    density = "rho"


CB_LABEL_MAP = {
    VelocityModelComponent.s_wave: "S Wave Velocity",
    VelocityModelComponent.p_wave: "P Wave Velocity",
    VelocityModelComponent.density: "Density",
}


def plot_velocity_model(
    fig: pygmt.Figure,
    box: BoundingBox,
    velocity_model: np.memmap,
    resolution: float,
    slice: int,
    nx: int,
    ny: int,
    region: tuple[float, ...],
    **kwargs,
):
    """Plot the velocity model on a pygmt figure.

    Parameters
    ----------
    fig : pygmt.Figure
        The figure to plot on.
    box : BoundingBox
        The bounding box the bounds of the velocity model.
    velocity_model : np.memmap
        The velocity model data as a memory-mapped array.
    resolution : float
        The resolution of the grid in meters.
    slice : int
        The z-level slice of the velocity model to plot.
    nx : int
        The number of grid points in the x-direction.
    ny : int
        The number of grid points in the y-direction.
    region : tuple of float
        The plotting region.
    **kwargs
        Additional keyword arguments passed to plotting.plot_grid.
    """
    corners = np.c_[box.corners, np.zeros_like(box.corners[:, 0])]

    # The indexing here is to account for:
    # 1. The coordinate meshgrid generating along the boundary of
    # the domain (hence, 1:-1, 1:-1),
    # 2. The fact we don't need depth (hence :2)
    lat_lon_grid = grid.coordinate_meshgrid(
        corners[-1], corners[-2], corners[0], resolution * 1000
    )[1:-1, 1:-1, :2]

    # The lat lon grid has shape (nx, ny, nz), so we flip that to
    # make the veloity model.
    lat_lon_grid = np.transpose(lat_lon_grid, (1, 0, 2))
    velocity_slice = velocity_model[:, slice, :].reshape((ny, nx))
    velocity_model_df = pd.DataFrame(
        {
            # Additionally, the lat lon grid is reversed compared to the velocity model.
            "lat": lat_lon_grid[::-1, ::-1, 0].ravel(),
            "lon": lat_lon_grid[::-1, ::-1, 1].ravel(),
            "value": velocity_slice.ravel(),
        }
    )

    velocity_slice = velocity_model[:, slice, :].reshape((ny, nx))
    velocity_model_df = pd.DataFrame(
        {
            # Additionally, the lat lon grid is reversed compared to the velocity model.
            "lat": lat_lon_grid[::-1, ::-1, 0].ravel(),
            "lon": lat_lon_grid[::-1, ::-1, 1].ravel(),
            "value": velocity_slice.ravel(),
        }
    )

    cur_grid = plotting.create_grid(
        velocity_model_df,
        "value",
        grid_spacing="100e/100e",
        region=region,
        set_water_to_nan=False,
    )
    cmap_limits = (
        velocity_model_df["value"].min().round(1),
        velocity_model_df["value"].max().round(1),
        (
            (velocity_model_df["value"].max() - velocity_model_df["value"].min()) / 10
        ).round(1),
    )

    plotting.plot_grid(
        fig,
        cur_grid,
        "hot",
        cmap_limits,
        ("white", "black"),
        reverse_cmap=True,
        plot_contours=False,
        continuous_cmap=True,
        **kwargs,
    )


def load_velocity_model_file(
    velocity_model_ffp: Path,
    component: VelocityModelComponent,
    nx: int,
    ny: int,
    nz: int,
) -> np.memmap:
    """Load a velocity model from a file as a memory-mapped array.

    Parameters
    ----------
    velocity_model_ffp : Path
        The file path to the velocity model data.
    component : VelocityModelComponent
        The component of the velocity model to load.
    nx : int
        The number of grid points in the x-direction.
    ny : int
        The number of grid points in the y-direction.
    nz : int
        The number of grid points in the z-direction.

    Returns
    -------
    np.memmap
        The loaded velocity model as a memory-mapped array.
    """
    filepath_map = {
        VelocityModelComponent.p_wave: "vp3dfile.p",
        VelocityModelComponent.s_wave: "vs3dfile.s",
        VelocityModelComponent.density: "rho3dfile.d",
    }

    return np.memmap(
        velocity_model_ffp / filepath_map[component],
        shape=(ny, nz, nx),
        # NOTE: This may require tweaks in the future to account
        # for differing endianness across machines. For now, this
        # should be fine.
        dtype="<f4",
    )


@app.command(
    help="Plot a velocity model from a velocity model params yaml file (vm_params.yaml).",
    name="vm-params",
)
def plot_vm_params(
    vm_params_ffp: Annotated[
        Path, typer.Argument(help="Path to velocity model parameters")
    ],
    output_plot_path: Annotated[
        Path,
        typer.Argument(
            help="Path to output velocity model plot.", writable=True, dir_okay=False
        ),
    ],
    velocity_model_ffp: Annotated[
        Path | None,
        typer.Option(
            help="Path to velocity model directory", file_okay=False, exists=True
        ),
    ] = None,
    component: Annotated[
        VelocityModelComponent,
        typer.Option(help="Velocity model component to overlay."),
    ] = VelocityModelComponent.p_wave,
    slice: Annotated[
        int, typer.Option(help="z-level slice of velocity model to plot.")
    ] = 0,
    transparency: Annotated[
        int, typer.Option(help="Velocity model overlay transparency, (0 = opaque)")
    ] = 80,
    title: Annotated[Optional[str], typer.Option(help="Figure title")] = None,
    latitude_pad: Annotated[
        float, typer.Option(help="Latitude padding for figure (in degrees)", min=0)
    ] = 0,
    longitude_pad: Annotated[
        float, typer.Option(help="Longitude padding for figure (in degrees)", min=0)
    ] = 0,
    dpi: Annotated[
        float, typer.Option(help="Plot output DPI (higher is better)")
    ] = 300,
) -> None:
    """Plot a velocity model from a velocity model params yaml file (vm_params.yaml).

    Parameters
    ----------
    vm_params_ffp : Path
        Path to the YAML file containing velocity model parameters.
    output_plot_path : Path
        Path to save the generated plot.
    velocity_model_ffp : Path | None
        Path to the directory containing velocity model files. If None, no velocity model is overlaid.
    component : VelocityModelComponent, default = `VelocityModelComponent.p_wave`
        The velocity model component to overlay.
    slice : int, optional, default = 0
        The z-level slice of the velocity model to plot.
    transparency : int, optional, default = 80
        Transparency level for the velocity model overlay.
    title : str, optional
        Title of the figure.
    latitude_pad : float, optional, default = 0
        Padding for latitude (in degrees) for the figure.
    longitude_pad : float, optional, default = 0
        Padding for longitude (in degrees) for the figure.
    dpi : float, optional, default = 300
        DPI for the output plot.
    """
    with open(vm_params_ffp) as vm_params_handle:
        vm_params = yaml.safe_load(vm_params_handle)

        # Extent x and extent y are swapped in meaning between the old
        # vm params and the new bounding box class. So we swap them
        # here when we load them.
        origin = np.array([vm_params["MODEL_LAT"], vm_params["MODEL_LON"]])
        great_circle_bearing = vm_params["MODEL_ROT"]
        extent_x = vm_params["extent_x"]
        extent_y = vm_params["extent_y"]
        nztm_bearing = coordinates.great_circle_bearing_to_nztm_bearing(
            origin, extent_y / 2, great_circle_bearing
        )
        box = bounding_box.BoundingBox.from_centroid_bearing_extents(
            origin, nztm_bearing, extent_y, extent_x
        )

        resolution = vm_params.get("hh")
        nx = vm_params["nx"]
        ny = vm_params["ny"]
        nz = vm_params["nz"]

    region = (
        box.corners[:, 1].min() - longitude_pad,
        box.corners[:, 1].max() + longitude_pad,
        box.corners[:, 0].min() - latitude_pad,
        box.corners[:, 0].max() + latitude_pad,
    )
    fig = plotting.gen_region_fig(title, region)
    if velocity_model_ffp:
        velocity_model = load_velocity_model_file(
            velocity_model_ffp, component, nx, ny, nz
        )
        plot_velocity_model(
            fig,
            box,
            velocity_model,
            resolution,
            slice,
            nx,
            ny,
            region,
            cb_label=CB_LABEL_MAP[component],
            transparency=transparency,
        )
    fig.plot(
        x=box.corners[:, 1],
        y=box.corners[:, 0],
        close=True,
        pen="1p,black,-",
    )
    fig.savefig(
        output_plot_path,
        dpi=dpi,
        anti_alias=True,
    )


if __name__ == "__main__":
    app()
