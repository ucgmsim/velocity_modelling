"""
3D Stacked Surface Viewer.

This script visualizes one or more digital elevation models (DEMs) or other
surfaces from HDF5 files as a stacked 3D plot using Plotly. It allows for
cropping to a specific geographic region, draping outlines from a GeoJSON
file over the top surface, and interactively exploring the stacked surfaces.
"""

import json
from pathlib import Path
from typing import Annotated, Optional

import h5py
import numpy as np
import plotly.graph_objects as go
import typer
from scipy.interpolate import RegularGridInterpolator

from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


def load_outline_coords(geojson_path: Path) -> list[np.ndarray]:
    """Return a list of Nx2 [lon, lat] numpy arrays from the GeoJSON file.

    Parses a GeoJSON file and extracts coordinate sequences from LineString,
    Polygon, MultiLineString, and MultiPolygon features.

    Parameters
    ----------
    geojson_path : Path
        Path to the input GeoJSON file.

    Returns
    -------
    list[np.ndarray]
        A list of numpy arrays, where each array contains [lon, lat] coordinates.
    """
    with open(geojson_path) as f:
        gj = json.load(f)

    outlines = []
    for feat in gj.get("features", []):
        geom = feat["geometry"]
        if geom["type"] in ("LineString", "Polygon"):
            # Polygon → use its outer ring
            coords = (
                geom["coordinates"][0]
                if geom["type"] == "Polygon"
                else geom["coordinates"]
            )
            outlines.append(np.asarray(coords))
        elif geom["type"] in ("MultiLineString", "MultiPolygon"):
            for part in geom["coordinates"]:
                coords = part[0] if geom["type"] == "MultiPolygon" else part
                outlines.append(np.asarray(coords))
    return outlines


def load_hdf5_data(
    file_paths: list[Path],
) -> dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load lat, lon, elevation from HDF5 surface files.

    Parameters
    ----------
    file_paths : list[Path]
        A list of paths to the HDF5 files.

    Returns
    -------
    dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]]
        A dictionary mapping each file path to a tuple containing the
        latitude, longitude, and elevation arrays.
    """
    return {
        path: (f["latitude"][:], f["longitude"][:], f["elevation"][:])
        for path in file_paths
        for f in [h5py.File(path, "r")]
    }


def filter_data(
    lat: np.ndarray,
    lon: np.ndarray,
    elevation: np.ndarray,
    lat_range: Optional[tuple[float, float]] = None,
    lon_range: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter data to a specified latitude/longitude range.

    Parameters
    ----------
    lat : np.ndarray
        Array of latitudes.
    lon : np.ndarray
        Array of longitudes.
    elevation : np.ndarray
        2D array of elevation values.
    lat_range : Optional[tuple[float, float]], optional
        Latitude range [min, max] for filtering, by default None.
    lon_range : Optional[tuple[float, float]], optional
        Longitude range [min, max] for filtering, by default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The filtered latitude, longitude, and elevation arrays.
    """
    lat_mask = (
        (lat >= lat_range[0]) & (lat <= lat_range[1])
        if lat_range
        else np.ones_like(lat, dtype=bool)
    )
    lon_mask = (
        (lon >= lon_range[0]) & (lon <= lon_range[1])
        if lon_range
        else np.ones_like(lon, dtype=bool)
    )
    return lat[lat_mask], lon[lon_mask], elevation[np.ix_(lat_mask, lon_mask)]


def plot_stacked_surfaces(
    data_dict: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]],
    crop_bounds: Optional[tuple[float, float, float, float]] = None,
    elev_range: Optional[tuple[float, float]] = None,
    outline_list: Optional[list[np.ndarray]] = None,
) -> None:
    """Render stacked DEM surfaces using Plotly.

    Parameters
    ----------
    data_dict : dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]]
        A mapping from file path to a tuple of (lat, lon, elevation) arrays.
        The first item is treated as the top surface.
    crop_bounds : Optional[tuple[float, float, float, float]], optional
        Optional lat/lon box (lat_min, lat_max, lon_min, lon_max) to which the
        TOP surface is cropped, by default None.
    elev_range : Optional[tuple[float, float]], optional
        A tuple [z_min, z_max] to lock the z-axis. If None, it is auto-detected,
        by default None.
    outline_list : Optional[list[np.ndarray]], optional
        A list of numpy arrays containing lon/lat outline vertices. Each outline
        is draped on the top surface in yellow, by default None.
    """

    # ---------- axis limits ----------
    if crop_bounds:
        lat_min, lat_max, lon_min, lon_max = crop_bounds
        lat_range_list = [lat_min, lat_max]
        lon_range_list = [lon_min, lon_max]
    else:
        # fall back to full extents from all layers
        lat_range_list = [
            min(v[0].min() for v in data_dict.values()),
            max(v[0].max() for v in data_dict.values()),
        ]
        lon_range_list = [
            min(v[1].min() for v in data_dict.values()),
            max(v[1].max() for v in data_dict.values()),
        ]

    # z-axis range (symmetric if diverging cmap)
    if elev_range is None:
        all_elev = np.concatenate([v[2].ravel() for v in data_dict.values()])
        elev_range = [all_elev.min(), all_elev.max()]
    z_min, z_max = elev_range
    vmax = max(abs(z_min), abs(z_max))
    vmin = -vmax

    # ---------- figure ----------
    fig = go.Figure()

    # ---------- loop over layers ----------
    for idx, (fname, (lat, lon, elev)) in enumerate(data_dict.items()):
        # crop *only* the top layer if crop_bounds requested
        if idx == 0 and crop_bounds:
            lat_mask = (lat >= lat_range_list[0]) & (lat <= lat_range_list[1])
            lon_mask = (lon >= lon_range_list[0]) & (lon <= lon_range_list[1])
            lat = lat[lat_mask]
            lon = lon[lon_mask]
            elev = elev[np.ix_(lat_mask, lon_mask)]

        lon_mesh, lat_mesh = np.meshgrid(lon, lat)

        fig.add_trace(
            go.Surface(
                z=elev,
                x=lon_mesh,
                y=lat_mesh,
                surfacecolor=elev,
                colorscale="RdBu",
                cmin=vmin,
                cmax=vmax,
                showscale=False,
                opacity=1.0 if idx == 0 else 0.5,
                name=str(fname),
                legendgroup=str(fname),
                showlegend=True,
            )
        )

    # ---------- optional outlines ----------
    if outline_list:
        # Build an interpolator on the (possibly-cropped) top surface
        top_lat, top_lon, top_elev = next(iter(data_dict.values()))
        if crop_bounds:
            lat_mask = (top_lat >= lat_range_list[0]) & (top_lat <= lat_range_list[1])
            lon_mask = (top_lon >= lon_range_list[0]) & (top_lon <= lon_range_list[1])
            top_lat = top_lat[lat_mask]
            top_lon = top_lon[lon_mask]
            top_elev = top_elev[np.ix_(lat_mask, lon_mask)]

        z_interp = RegularGridInterpolator(
            (top_lat, top_lon), top_elev, bounds_error=False, fill_value=np.nan
        )
        for arr in outline_list:  # arr shape (N,2) → lon,lat
            lon_line, lat_line = arr[:, 0], arr[:, 1]
            z_line = z_interp(np.column_stack([lat_line, lon_line])) + 1.0  # lift 1 m
            fig.add_trace(
                go.Scatter3d(
                    x=lon_line,
                    y=lat_line,
                    z=z_line,
                    mode="lines",
                    line=dict(color="yellow", width=4),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # ---------- layout ----------
    fig.update_layout(
        title="Stacked 3-D Terrain Surfaces",
        scene=dict(
            xaxis=dict(title="Longitude", range=lon_range_list),
            yaxis=dict(title="Latitude", range=lat_range_list),
            zaxis=dict(title="Elevation", range=[-vmax, vmax]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.10),
        ),
        legend=dict(title="Toggle layers", itemsizing="constant"),
        hoverlabel=dict(font_size=11, font_family="Arial", namelength=-1),
    )

    fig.show()


@cli.from_docstring(app)
def main(
    files: Annotated[
        list[Path],
        typer.Argument(
            help="List of HDF5 files to process.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    lat_range: Annotated[
        Optional[tuple[float, float]],
        typer.Option(help="Latitude range [min max] to crop data."),
    ] = None,
    lon_range: Annotated[
        Optional[tuple[float, float]],
        typer.Option(help="Longitude range [min max] to crop data."),
    ] = None,
    outline: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to a GeoJSON file whose outline will be draped on the top surface.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
):
    """Visualize stacked terrain surfaces with transparency and optional outlines.

    Parameters
    ----------
    files : list[Path]
        List of HDF5 files containing lat, lon, and elevation data.
    lat_range : Optional[tuple[float, float]]
        Latitude range [min, max] to crop the data. If None, no cropping is applied.
    lon_range : Optional[tuple[float, float]]
        Longitude range [min, max] to crop the data. If None, no cropping is applied.
    outline : Optional[Path]
        Path to a GeoJSON file whose outline will be draped on the top surface.

    """
    data_dict = load_hdf5_data(files)
    all_elevations = []
    for k in data_dict:
        lat, lon, elev = data_dict[k]
        lat, lon, elev = filter_data(lat, lon, elev, lat_range, lon_range)
        data_dict[k] = (lat, lon, elev)
        all_elevations.append(elev)

    # Determine elevation range from lower layers for consistent color scale
    if len(all_elevations) > 1:
        all_elev = np.concatenate([e.ravel() for e in all_elevations[1:]])
        elev_range = (all_elev.min(), all_elev.max())
    else:
        elev_range = None

    # Determine crop bounds from the combined extent of lower layers
    if len(data_dict) > 1:
        lower_bounds = list(data_dict.values())[1:]
        lower_lat = np.concatenate([v[0] for v in lower_bounds])
        lower_lon = np.concatenate([v[1] for v in lower_bounds])
        crop_bounds = (
            lower_lat.min(),
            lower_lat.max(),
            lower_lon.min(),
            lower_lon.max(),
        )
    else:
        crop_bounds = None

    # Optional outline
    outline_list = load_outline_coords(outline) if outline else None

    plot_stacked_surfaces(
        data_dict,
        crop_bounds=crop_bounds,
        elev_range=elev_range,
        outline_list=outline_list,
    )


if __name__ == "__main__":
    app()
