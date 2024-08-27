from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml

from pygmt_helper import plotting
from velocity_modelling import bounding_box

app = typer.Typer()


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
    realisation_ffp: Annotated[
        Optional[Path],
        typer.Option(help="Path to realisation JSON file", dir_okay=False),
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
):
    """Plot the boundary of the velocity model."""
    if vm_params_ffp is not None:
        with open(vm_params_ffp) as vm_params_handle:
            vm_params = yaml.safe_load(vm_params_handle)
        box = bounding_box.BoundingBox.from_centroid_bearing_extents(
            [vm_params["MODEL_LAT"], vm_params["MODEL_LON"]],
            vm_params["MODEL_ROT"],
            vm_params["extent_x"],
            vm_params["extent_y"],
        )
    elif realisation_ffp is not None:
        from workflow.realisations import DomainParameters, SourceConfig

        domain_parameters = DomainParameters.read_from_realisation(realisation_ffp)
        box = domain_parameters.domain

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
    else:
        print(
            "You must provide either a vm_params.yaml, realisation json file, or manually specify the domain parameters."
        )
        typer.Exit(code=1)

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
    if realisation_ffp is not None:
        source_config = SourceConfig.read_from_realisation(realisation_ffp)
        for fault in source_config.source_geometries.values():
            for plane in fault.corners.reshape((-1, 4, 3)):
                fig.plot(
                    x=plane[:, 1],
                    y=plane[:, 0],
                    close=True,
                    pen="0.5p,black",
                )

    fig.savefig(
        output_plot_path,
        dpi=dpi,
        anti_alias=True,
    )


if __name__ == "__main__":
    app()
