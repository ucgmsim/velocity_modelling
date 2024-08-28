from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
import yaml

from pygmt_helper import plotting
from qcore import grid
from velocity_modelling import bounding_box

app = typer.Typer()


class VelocityModelComponent(str, Enum):
    p_wave = "p_wave"
    s_wave = "s_wave"
    density = "rho"


@app.command()
def plot_velocity_model(
    output_plot_path: Annotated[
        Path,
        typer.Argument(
            help="Path to output velocity model plot.", writable=True, dir_okay=False
        ),
    ],
    vm_params_ffp: Annotated[
        Optional[Path], typer.Option(help="Path to VM params YAML", dir_okay=False)
    ] = None,
    velocity_model_ffp: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to velocity model directory", file_okay=False, exists=True
        ),
    ] = None,
    resolution: Annotated[
        Optional[float], typer.Option(help="Resolution of the velocity model")
    ] = None,
    centre_lat: Annotated[
        Optional[float], typer.Option(help="Centre latitude", min=-90, max=90)
    ] = None,
    centre_lon: Annotated[
        Optional[float], typer.Option(help="Centre longitude", min=-180, max=180)
    ] = None,
    bearing: Annotated[
        Optional[float], typer.Option(help="Model bearing", min=0, max=360)
    ] = None,
    extent_x: Annotated[
        Optional[float], typer.Option(help="Extent in x-direction", min=0)
    ] = None,
    extent_y: Annotated[
        Optional[float], typer.Option(help="Extent in y-direction", min=0)
    ] = None,
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
    component: Annotated[
        VelocityModelComponent,
        typer.Option(help="Velocity model component to overlay."),
    ] = VelocityModelComponent.p_wave,
    slice: Annotated[
        int, typer.Option(help="z-level slice of velocity model to plot.")
    ] = 0,
    nz: Annotated[
        Optional[int], typer.Option(help="Number of z-slices in velocity model")
    ] = None,
    transparency: Annotated[
        int, typer.Option(help="Velocity model overlay transparency, (0 = opaque)")
    ] = 80,
):
    """Plot the boundary of the velocity model."""
    nx = ny = None
    if vm_params_ffp:
        with open(vm_params_ffp) as vm_params_handle:
            vm_params = yaml.safe_load(vm_params_handle)

        # Extent x and extent y are swapped in meaning between the old
        # vm params and the new bounding box class. So we swap them
        # here when we load them.
        box = bounding_box.BoundingBox.from_centroid_bearing_extents(
            [vm_params["MODEL_LAT"], vm_params["MODEL_LON"]],
            vm_params["MODEL_ROT"],
            vm_params["extent_y"],
            vm_params["extent_x"],
        )

        resolution = vm_params.get("hh")
        nx = vm_params.get("nx")
        ny = vm_params.get("ny")
        nz = vm_params.get("nz")

    elif all(
        parameter is not None
        for parameter in [
            centre_lat,
            centre_lon,
            bearing,
            extent_x,
            extent_y,
        ]
    ):
        box = bounding_box.BoundingBox.from_centroid_bearing_extents(
            [centre_lat, centre_lon], bearing, extent_x, extent_y
        )
        if resolution:
            nx = int(round(extent_x / resolution))
            ny = int(round(extent_y / resolution))
    else:
        print(
            "You must provide either a vm_params.yaml file, or manually specify the domain parameters."
        )
        raise typer.Exit(code=1)
    region = (
        box.corners[:, 1].min() - longitude_pad,
        box.corners[:, 1].max() + longitude_pad,
        box.corners[:, 0].min() - latitude_pad,
        box.corners[:, 0].max() + latitude_pad,
    )
    fig = plotting.gen_region_fig(title, region)
    fig.plot(
        x=box.corners[:, 1],
        y=box.corners[:, 0],
        close=True,
        pen="1p,black,-",
    )

    if velocity_model_ffp and not resolution:
        print(
            "You must provide a resolution for the velocity model if you supply a velocity model."
        )
        raise typer.Exit(1)
    elif velocity_model_ffp:
        filepath_map = {
            VelocityModelComponent.p_wave: "vp3dfile.p",
            VelocityModelComponent.s_wave: "vs3dfile.s",
            VelocityModelComponent.density: "rho3dfile.d",
        }

        velocity_model = np.memmap(
            velocity_model_ffp / filepath_map[component],
            shape=(ny, nz, nx),
            # NOTE: This may require tweaks in the future to account
            # for differing endianness across machines. For now, this
            # should be fine.
            dtype="<f4",
        )

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
                (velocity_model_df["value"].max() - velocity_model_df["value"].min())
                / 10
            ).round(1),
        )
        print(cmap_limits)
        cb_label_map = {
            VelocityModelComponent.s_wave: "S Wave Velocity",
            VelocityModelComponent.p_wave: "P Wave Velocity",
            VelocityModelComponent.density: "Density",
        }
        plotting.plot_grid(
            fig,
            cur_grid,
            "hot",
            cmap_limits,
            ("white", "black"),
            transparency=transparency,
            reverse_cmap=True,
            plot_contours=False,
            cb_label=cb_label_map[component],
            continuous_cmap=True,
        )

    fig.savefig(
        output_plot_path,
        dpi=dpi,
        anti_alias=True,
    )


if __name__ == "__main__":
    app()
