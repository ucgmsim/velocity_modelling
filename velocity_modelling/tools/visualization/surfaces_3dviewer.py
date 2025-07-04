import argparse
import h5py
import numpy as np
import plotly.graph_objects as go
import requests
from io import BytesIO
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

import json

def load_outline_coords(geojson_path):
    """Return a list of Nx2 [lon, lat] numpy arrays from the GeoJSON file."""
    with open(geojson_path) as f:
        gj = json.load(f)

    outlines = []
    for feat in gj.get("features", []):
        geom = feat["geometry"]
        if geom["type"] in ("LineString", "Polygon"):
            # Polygon → use its outer ring
            coords = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"]
            outlines.append(np.asarray(coords))
        elif geom["type"] in ("MultiLineString", "MultiPolygon"):
            for part in geom["coordinates"]:
                coords = part[0] if geom["type"] == "MultiPolygon" else part
                outlines.append(np.asarray(coords))
    return outlines



def load_hdf5_data(file_paths):
    """Load lat, lon, elevation from HDF5 surface files."""
    return {
        path: (
            f['latitude'][:],
            f['longitude'][:],
            f['elevation'][:]
        ) for path in file_paths
        for f in [h5py.File(path, 'r')]
    }

def filter_data(lat, lon, elevation, lat_range=None, lon_range=None):
    """Filter data to lat/lon range."""
    lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1]) if lat_range else np.ones_like(lat, dtype=bool)
    lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1]) if lon_range else np.ones_like(lon, dtype=bool)
    return lat[lat_mask], lon[lon_mask], elevation[np.ix_(lat_mask, lon_mask)]


def plot_stacked_surfaces(
    data_dict: dict,
    crop_bounds: tuple | None = None,           # (lat_min, lat_max, lon_min, lon_max)
    elev_range: list | tuple | None = None,     # [z_min, z_max]
    outline_list: list[np.ndarray] | None = None   # list of (N,2) arrays [lon,lat]
) -> None:
    """
    Render stacked DEM surfaces.

    Parameters
    ----------
    data_dict      mapping: file_name → (lat, lon, elevation)
                   *first* item is treated as the top surface.
    crop_bounds    optional lat/lon box to which the TOP surface is cropped.
    elev_range     [z_min, z_max] to lock the z-axis (pass the global range of lower layers).
    outline_list   list of numpy arrays containing lon/lat outline vertices.  Each outline
                   is draped on the top surface in yellow.
    """

    # ---------- axis limits ----------
    if crop_bounds:
        lat_min, lat_max, lon_min, lon_max = crop_bounds
        lat_range = [lat_min, lat_max]
        lon_range = [lon_min, lon_max]
    else:
        # fall back to full extents from all layers
        lat_range = [
            min(v[0].min() for v in data_dict.values()),
            max(v[0].max() for v in data_dict.values())
        ]
        lon_range = [
            min(v[1].min() for v in data_dict.values()),
            max(v[1].max() for v in data_dict.values())
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
            lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
            lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])
            lat  = lat[lat_mask]
            lon  = lon[lon_mask]
            elev = elev[np.ix_(lat_mask, lon_mask)]

        lon_mesh, lat_mesh = np.meshgrid(lon, lat)

        fig.add_trace(go.Surface(
            z=elev,
            x=lon_mesh,
            y=lat_mesh,
            surfacecolor=elev,
            colorscale="RdBu",
            cmin=vmin, cmax=vmax,
            showscale=False,
            opacity=1.0 if idx == 0 else 0.5,
            name=fname,
            legendgroup=fname,
            showlegend=True,
        ))

    # ---------- optional outlines ----------
    if outline_list:
        # Build an interpolator on the (possibly-cropped) top surface
        top_lat, top_lon, top_elev = next(iter(data_dict.values()))
        z_interp = RegularGridInterpolator(
            (top_lat, top_lon), top_elev, bounds_error=False, fill_value=np.nan
        )
        for arr in outline_list:                      # arr shape (N,2) → lon,lat
            lon_line, lat_line = arr[:, 0], arr[:, 1]
            z_line = z_interp(np.column_stack([lat_line, lon_line])) + 1.0  # lift 1 m
            fig.add_trace(go.Scatter3d(
                x=lon_line,
                y=lat_line,
                z=z_line,
                mode="lines",
                line=dict(color="yellow", width=4),
                hoverinfo="skip",
                showlegend=False
            ))

    # ---------- layout ----------
    fig.update_layout(
        title="Stacked 3-D Terrain Surfaces",
        scene=dict(
            xaxis=dict(title="Longitude", range=lon_range),
            yaxis=dict(title="Latitude",  range=lat_range),
            zaxis=dict(title="Elevation", range=[-vmax, vmax]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.10),
        ),
        legend=dict(title="Toggle layers", itemsizing="constant"),
        hoverlabel=dict(font_size=11, font_family="Arial", namelength=-1),
    )

    fig.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize stacked terrain surfaces with transparency toggles and color legend.")
    parser.add_argument("files", nargs="+", help="List of HDF5 files to process.")
    parser.add_argument("--lat_range", nargs=2, type=float, default=[-45, -42])
    parser.add_argument("--lon_range", nargs=2, type=float, default=[170, 174])
    parser.add_argument(
        "--outline",
        type=str,
        default=None,
        help="Path to a GeoJSON file whose LineString/Polygon outline "
             "will be draped in yellow on the top surface"
    )
    args = parser.parse_args()

    data_dict = load_hdf5_data(args.files)
    all_elevations = []
    for k in data_dict:
        lat, lon, elev = data_dict[k]
        lat, lon, elev = filter_data(lat, lon, elev, args.lat_range, args.lon_range)
        data_dict[k] = (lat, lon, elev)
        all_elevations.append(elev)

    all_elev = np.concatenate([e.ravel() for e in all_elevations[1:]])
    elev_range = [all_elev.min(), all_elev.max()]
    vmax = np.max(np.abs(all_elev))

    # Use top layer bounds to fetch basemap
    lat0, lon0, _ = data_dict[args.files[0]]
    lat_min, lat_max = lat0.min(), lat0.max()
    lon_min, lon_max = lon0.min(), lon0.max()

    lon_mesh0, lat_mesh0 = np.meshgrid(lon0, lat0)


    # Crop bounds from lower layers
    lower_bounds = list(data_dict.values())[1:]
    lower_lat = np.concatenate([v[0] for v in lower_bounds])
    lower_lon = np.concatenate([v[1] for v in lower_bounds])
    crop_bounds = (lower_lat.min(), lower_lat.max(), lower_lon.min(), lower_lon.max())

    # Optional outline
    outline_list = load_outline_coords(args.outline) if args.outline else None

    plot_stacked_surfaces(
        data_dict,
        crop_bounds=crop_bounds,
        elev_range=elev_range,
        outline_list=outline_list
    )


