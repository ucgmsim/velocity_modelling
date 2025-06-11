#!/usr/bin/env python3
"""
Extract cross-section from a velocity model and plot it.
This script reads a velocity model from an HDF5 file and generates a cross-section plot
based on specified coordinates or grid indices. It also allows for saving the cross-section data
to a CSV file.

Usage:
    python extract_cross_section.py <h5file> [options]
    --lat1 <lat1> --lon1 <lon1> --lat2 <lat2> --lon2 <lon2>
    --x1 <x1> --y1 <y1> --x2 <x2> --y2 <y2>
    --property <property_name> --xaxis <xaxis> --n_points <n_points>
    --vmin <vmin> --vmax <vmax> --max_depth <max_depth>
    --cmap <cmap> --png <output_png> --csv <csv_output>

    --help
    -h, --help  Show this help message and exit.
    <h5file>       Path to the HDF5 file containing the velocity model.
    --lat1 <lat1>   Latitude of the first point of the cross-section.
    --lon1 <lon1>   Longitude of the first point of the cross-section.
    --lat2 <lat2>   Latitude of the second point of the cross-section.
    --lon2 <lon2>   Longitude of the second point of the cross-section.
    --x1 <x1>       Grid index of the first point of the cross-section.
    --y1 <y1>       Grid index of the first point of the cross-section.
    --x2 <x2>       Grid index of the second point of the cross-section.
    --y2 <y2>       Grid index of the second point of the cross-section.
    --property <property_name>
                    Property to extract from the velocity model (default: "vp").
    --xaxis <xaxis> Choose x-axis for cross-section ("auto", "lat", "lon").

    --n_points <n_points>
                    Number of points in the cross-section (default: 350).
    --vmin <vmin>   Minimum value for color scaling (default: None).
    --vmax <vmax>   Maximum value for color scaling (default: None).
    --max_depth <max_depth>
                    Maximum depth to include in the cross-section (default: None).
    --cmap <cmap>   Colormap for the cross-section (default: "jet").
    --png <output_png>
                    Prefix to save the output PNG image (default: None). If specified,
                    two images will be created: <output_png>_plot.png and
                    <output_png>_map.png.
    --csv <csv_output>
                    Path to save the cross-section data as CSV (default: None).
    --help         Show this help message and exit.


"""

from pathlib import Path
from typing import Annotated, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


@cli.from_docstring(app)
def extract_cross_section(
    h5file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    lat1: Annotated[Optional[float], typer.Option()] = None,
    lon1: Annotated[Optional[float], typer.Option()] = None,
    lat2: Annotated[Optional[float], typer.Option()] = None,
    lon2: Annotated[Optional[float], typer.Option()] = None,
    x1: Annotated[Optional[int], typer.Option()] = None,
    y1: Annotated[Optional[int], typer.Option()] = None,
    x2: Annotated[Optional[int], typer.Option()] = None,
    y2: Annotated[Optional[int], typer.Option()] = None,
    property_name: Annotated[str, typer.Option()] = "vp",
    xaxis: Annotated[str, typer.Option()] = "auto",
    n_points: Annotated[int, typer.Option()] = 350,
    vmin: Annotated[Optional[float], typer.Option()] = None,
    vmax: Annotated[Optional[float], typer.Option()] = None,
    max_depth: Annotated[Optional[float], typer.Option()] = None,
    cmap: Annotated[str, typer.Option()] = "jet",
    png: Annotated[Optional[Path], typer.Option()] = None,
    csv: Annotated[Optional[Path], typer.Option()] = None,
):
    """
    Plot a cross-section of velocity model.

    Parameters
    ----------
    h5file : Path
        Path to the HDF5 file containing the velocity model.
    lat1, lon1 : float, optional
        Latitude and longitude of the first point of the cross-section.
    lat2, lon2 : float, optional
        Latitude and longitude of the second point of the cross-section.
    x1, y1 : int, optional
        Grid indices of the first point of the cross-section.
    x2, y2 : int, optional
        Grid indices of the second point of the cross-section.
    property_name : str, optional
        Property to extract (default: "vp").
    xaxis : str, optional
        X-axis type for the cross-section ("auto", "lat", "lon").
    n_points : int, optional
        Number of points in the cross-section (default: 350).
    vmin, vmax : float, optional
        Minimum and maximum values for color scaling.
    max_depth : float, optional
        Maximum depth to include in the cross-section.
    cmap : str, optional
        Colormap for the cross-section (default: "jet").
    png : Path, optional
        Prefix to save the output PNG image, if specified, xxx_plot.png and xxx_map.png will be created.
    csv : Path, optional
        Path to save the cross-section data as CSV.
    """

    # Raise ValueError if both grid indices and coordinates are specified
    if (x1 is not None and lon1 is not None) or (y1 is not None and lat1 is not None):
        raise ValueError(
            "Both grid indices (x1, y1) and coordinates (lat1, lon1) are specified. "
            "Please specify only one of them."
        )

    print(f"Creating improved cross-section from {h5file}")
    with h5py.File(h5file, "r") as f:
        lat = f["/mesh/lat"][()]
        lon = f["/mesh/lon"][()]
        z = f["/mesh/z"][()]
        data = f[f"/properties/{property_name}"][()]

        config = f.get("/config")
        if not isinstance(config, h5py.Group):
            raise KeyError("Missing required /config group in HDF5 file.")

        # Required attributes, including h_depth now
        required_attrs = ["origin_lat", "origin_lon", "origin_rot", "h_depth"]
        missing = [attr for attr in required_attrs if attr not in config.attrs]
        if missing:
            raise KeyError(
                f"Missing required config attribute(s): {', '.join(missing)}"
            )

        try:
            origin_lat = float(config.attrs["origin_lat"])
            origin_lon = float(config.attrs["origin_lon"])
            origin_rot = float(config.attrs["origin_rot"])
            extent_z_spacing = float(config.attrs["h_depth"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value in config attributes: {e}")

        z_km = z * extent_z_spacing

        if z[0] > z[-1]:
            z = z[::-1]
            z_km = z_km[::-1]
            data = data[::-1, :, :]

        if max_depth is not None:
            z_mask = z_km <= max_depth
            z = z[z_mask]
            z_km = z_km[z_mask]
            data = data[z_mask, :, :]

        # Derive coordinates if only lat/lon are given
        if lat1 is not None and lon1 is not None:
            (
                lat1_val,
                lon1_val,
            ) = lat1, lon1
            lat2_val = (
                lat2 if lat2 is not None else lat1
            )  # if undefined, use lat1. Horizontal cross-section
            lon2_val = (
                lon2 if lon2 is not None else lon1
            )  # if undefined, use lon1. Vertical cross-section
            dist1 = (lat - lat1_val) ** 2 + (lon - lon1_val) ** 2
            dist2 = (lat - lat2_val) ** 2 + (lon - lon2_val) ** 2
            x1, y1 = np.unravel_index(
                np.argmin(dist1), dist1.shape
            )  # find the closest point from (lat1,lon1)
            x2, y2 = np.unravel_index(
                np.argmin(dist2), dist2.shape
            )  # find the closest point from (lat2,lon2)
        elif x1 is not None and y1 is not None:
            x1 = int(x1)
            y1 = int(y1)
            x2 = (
                int(x2) if x2 is not None else x1
            )  # if undefined, use x1. Horizontal cross-section
            y2 = (
                int(y2) if y2 is not None else y1
            )  # if undefined, use y1. Vertical cross-section
            lat1_val = lat[x1, y1]
            lon1_val = lon[x1, y1]
            lat2_val = lat[x2, y2]
            lon2_val = lon[x2, y2]
        else:
            raise ValueError(
                "You must specify either --x1, --y1[, --x2, --y2] or --lat1, --lon1, --lat2, --lon2"
            )

        if lat1_val == lat2_val:
            # Horizontal cross-section (constant latitude)
            lat_pts = np.full(n_points, lat1_val)
            lon_pts = np.linspace(lon1_val, lon2_val, n_points)
        elif lon1_val == lon2_val:
            # Vertical cross-section (constant longitude)
            lat_pts = np.linspace(lat1_val, lat2_val, n_points)
            lon_pts = np.full(n_points, lon1_val)
        else:
            # Diagonal cross-section
            lat_pts = np.linspace(lat1_val, lat2_val, n_points)
            lon_pts = np.linspace(lon1_val, lon2_val, n_points)

        dx = lon2_val - lon1_val
        dy = lat2_val - lat1_val
        # Calculate the angle of the cross-section
        angle_deg = np.degrees(np.arctan2(dy, dx)) % 180

        # Determine the x-axis based on the angle
        if xaxis == "auto":
            xaxis = "lat" if angle_deg > 45 else "lon"

        if xaxis == "lat" and abs(dy) < 1e-6:
            raise ValueError(
                "Latitude cross-section requested but latitude range is zero."
            )
        if xaxis == "lon" and abs(dx) < 1e-6:
            raise ValueError(
                "Longitude cross-section requested but longitude range is zero."
            )

        cross_section = np.zeros((len(z), n_points))
        idx_map = []

        # Loop through each point along the cross-section path
        for i in range(n_points):
            # Calculate the distance from the current point to all grid points
            dist = (lat - lat_pts[i]) ** 2 + (lon - lon_pts[i]) ** 2
            # Find the grid point closest to the current point
            x_idx, y_idx = np.unravel_index(np.argmin(dist), dist.shape)
            # Extract the property values along the depth for the closest grid point
            for j, zval in enumerate(z):
                cross_section[j, i] = data[j, y_idx, x_idx]
            # Store the grid indices for reference
            idx_map.append((x_idx, y_idx))

        if csv:
            # Prepare data for DataFrame
            data_rows = []
            for i in range(n_points):
                x_idx, y_idx = idx_map[i]
                for j in range(len(z)):
                    data_rows.append(
                        {
                            "z_km": z_km[j],
                            "lon": lon[x_idx, y_idx],
                            "lat": lat[x_idx, y_idx],
                            "value": cross_section[j, i],
                            "x": x_idx,
                            "y": y_idx,
                        }
                    )

            # Create DataFrame and save to CSV
            df = pd.DataFrame(data_rows)
            df.to_csv(csv, index=False)

        # Instead of a combined figure, create two separate figures

        # First figure - Map
        fig_map = plt.figure(figsize=(8, 8))

        projection = ccrs.PlateCarree()
        ax_map = plt.axes(projection=projection)
        ax_map.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])

        # Turn off axes completely to remove the frame
        ax_map.axis("off")

        # Add features with adjusted styling
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="gray")
        ax_map.add_feature(
            cfeature.BORDERS, linestyle=":", linewidth=0.3, edgecolor="gray"
        )
        ax_map.add_feature(cfeature.LAND, edgecolor="none", facecolor="whitesmoke")
        ax_map.add_feature(cfeature.OCEAN, edgecolor="none", facecolor="lightcyan")
        ax_map.add_feature(
            cfeature.LAKES, edgecolor="gray", linewidth=0.3, facecolor="lightcyan"
        )
        ax_map.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="lightskyblue")

        # Add gridlines separately after turning off the axis frame
        gl = ax_map.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle=":"
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Plot domain outline and cross-section path
        domain_outline = [
            (lon[0, 0], lat[0, 0]),
            (lon[-1, 0], lat[-1, 0]),
            (lon[-1, -1], lat[-1, -1]),
            (lon[0, -1], lat[0, -1]),
            (lon[0, 0], lat[0, 0]),
        ]
        lons, lats = zip(*domain_outline)

        ax_map.plot(lons, lats, "r--", transform=projection, label="Domain")
        ax_map.plot(lon_pts, lat_pts, "b-", transform=projection, label="Cross-section")
        ax_map.plot(lon1_val, lat1_val, "bo", transform=projection)
        ax_map.plot(lon2_val, lat2_val, "bo", transform=projection)
        ax_map.text(lon1_val + 0.1, lat1_val, f"({lat1_val:.2f}, {lon1_val:.2f})")
        ax_map.text(lon2_val + 0.1, lat2_val, f"({lat2_val:.2f}, {lon2_val:.2f})")
        ax_map.plot(origin_lon, origin_lat, "ro", transform=projection, label="Origin")
        ax_map.text(
            origin_lon + 0.1,
            origin_lat,
            f"Rot: {origin_rot:.2f}°",
            transform=projection,
        )
        ax_map.legend(fontsize=8)
        ax_map.set_title("Velocity Model Domain with Cross-section Path", fontsize=12)

        # Save map if png is specified
        if png:
            map_filename = f"{png}_map.png"
            fig_map.savefig(map_filename, dpi=300, bbox_inches="tight")
            print(f"Map saved to {map_filename}")

        # Second figure - Cross-section
        fig_section = plt.figure(figsize=(12, 6))
        ax_section = plt.axes()

        x_vals = lon_pts if xaxis == "lon" else lat_pts
        xlabel = "Longitude" if xaxis == "lon" else "Latitude"
        ax_section.set_xlabel(xlabel)
        ax_section.set_ylabel("Depth (km)")
        ax_section.set_ylim([z_km.max(), z_km.min()])
        ax_section.tick_params(axis="x", direction="inout", top=True, which="both")
        ax_section.tick_params(axis="y", direction="inout", right=True, which="both")
        ax_section.minorticks_on()
        ax_section.xaxis.set_ticks(np.linspace(x_vals.min(), x_vals.max(), 11))
        ax_section.yaxis.set_ticks(np.linspace(z_km.min(), z_km.max(), 10))
        pcm = ax_section.pcolormesh(
            x_vals, z_km, cross_section, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )
        unit = "(g/cm³)" if property_name == "rho" else "(km/s)"
        fig_section.colorbar(pcm, ax=ax_section, label=f"{property_name} {unit}")
        ax_section.set_title(
            f"{property_name} : cross-section along {xlabel}\n"
            f"({lat1_val:.4f}, {lon1_val:.4f}) to ({lat2_val:.4f}, {lon2_val:.4f})\n"
            f"Index: x=({x1},{x2}), y=({y1},{y2})",
            fontsize=12,
        )

        # Save cross-section if png is specified
        if png:
            section_filename = f"{png}_plot.png"
            fig_section.savefig(section_filename, dpi=300, bbox_inches="tight")
            print(f"Cross-section saved to {section_filename}")

        # Display both figures
        plt.show()


if __name__ == "__main__":
    app()
