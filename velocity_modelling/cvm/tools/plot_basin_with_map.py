import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


# Custom tile class for Esri World Imagery
class EsriWorldImageryTiles(cimgt.GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        return url


def load_basement(file_path: Path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: Basement file {file_path} does not exist.")
        return None, None, None
    try:
        with file_path.open("r") as f:
            lines = f.readlines()

        n_lat, n_lon = map(int, lines[0].split())
        lats = np.array([float(x) for x in lines[1].split()])
        lons = np.array([float(x) for x in lines[2].split()])

        raster_lines = lines[3:]
        if len(raster_lines) != n_lat:
            print(
                f"Alert: Expected {n_lat} raster lines in {file_path}, got {len(raster_lines)}"
            )
            if len(raster_lines) < n_lat:
                raster_lines.extend(["0 " * n_lon] * (n_lat - len(raster_lines)))
            raster_lines = raster_lines[:n_lat]

        raster = []
        for lat_idx, line in enumerate(raster_lines):
            values = [float(x) for x in line.split()]
            if len(values) != n_lon:
                print(
                    f"Alert: Expected {n_lon} values in line {lat_idx + 4} of {file_path}, got {len(values)}"
                )
                if len(values) < n_lon:
                    values.extend([0] * (n_lon - len(values)))
                values = values[:n_lon]
            raster.append(values)

        raster = np.array(raster)
        if len(raster_lines) != n_lat or any(
            len(line.split()) != n_lon for line in raster_lines
        ):
            with file_path.open("w") as f:
                f.write(f"{n_lat} {n_lon}\n")
                f.write(" ".join(map(str, lats)) + "\n")
                f.write(" ".join(map(str, lons)) + "\n")
                for row in raster:
                    f.write(" ".join(map(str, row)) + "\n")

        return lats, lons, raster

    except Exception as e:
        print(f"Error loading basement file {file_path}: {e}")
        return None, None, None


def load_boundary(file_path: Path, is_boundary=True):
    file_path = Path(file_path)
    if not file_path.exists():
        print(
            f"Error: {'Boundary' if is_boundary else 'Smoothing'} file {file_path} does not exist."
        )
        return None, None
    try:
        with file_path.open("r") as f:
            lines = f.readlines()

        coords = [list(map(float, line.split())) for line in lines]
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]

        if is_boundary and coords[0] != coords[-1]:
            print(f"Alert: Boundary in {file_path} is not closed. Closing it.")
            coords.append(coords[0])
            lons.append(lons[0])
            lats.append(lats[0])
            with file_path.open("w") as f:
                for lon, lat in coords:
                    f.write(f"{lon} {lat}\n")

        return lons, lats

    except Exception as e:
        print(
            f"Error loading {'boundary' if is_boundary else 'smoothing'} file {file_path}: {e}"
        )
        return None, None


def plot_data(
    basement_file: Path,
    boundary_files: list[Path],
    smoothing_file: Path,
    basin_name: str,
    out_dir: Path,
):
    # Load basement data
    lats, lons, raster = load_basement(basement_file)
    if lats is None:
        return

    # Load all boundary files
    boundaries = []
    for boundary_file in boundary_files:
        boundary_lons, boundary_lats = load_boundary(boundary_file, True)
        if boundary_lons is not None:
            boundaries.append((boundary_lons, boundary_lats))
    if not boundaries:
        print("No valid boundary files loaded. Exiting.")
        return

    # Load smoothing data if provided
    smoothing_lons, smoothing_lats = None, None
    if smoothing_file:
        smoothing_lons, smoothing_lats = load_boundary(smoothing_file, False)

    # Calculate dynamic extent with 40% buffer
    all_lons = list(lons)
    all_lats = list(lats)
    for boundary_lons, boundary_lats in boundaries:
        all_lons.extend(boundary_lons)
        all_lats.extend(boundary_lats)
    if smoothing_lons and smoothing_lats:
        all_lons.extend(smoothing_lons)
        all_lats.extend(smoothing_lats)

    lon_min, lon_max = min(all_lons), max(all_lons)
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_buffer = (lon_max - lon_min) * 0.3 or 0.5
    lat_buffer = (lat_max - lat_min) * 0.2 or 0.5
    extent = [
        lon_min - lon_buffer,
        lon_max + lon_buffer,
        lat_min - lat_buffer,
        lat_max + lat_buffer,
    ]
    print(f"Map extent: {extent}")

    # Create map
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Add Esri World Imagery basemap using custom tile class
    esri_imagery = EsriWorldImageryTiles()
    ax.add_image(esri_imagery, 12)  # Zoom level 12 for high detail

    # Add additional features for context
    # ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)
    # ax.add_feature(cfeature.LAKES, edgecolor='blue', facecolor='lightblue', alpha=0.5)
    # ax.add_feature(cfeature.RIVERS, edgecolor='lightblue', alpha=0.5)

    # Add town names
    shp = shpreader.natural_earth(
        resolution="10m", category="cultural", name="populated_places"
    )
    reader = shpreader.Reader(shp)
    places = list(reader.records())
    for place in places:
        lon, lat = place.geometry.x, place.geometry.y
        if (extent[0] <= lon <= extent[1]) and (extent[2] <= lat <= extent[3]):
            ax.text(
                lon,
                lat,
                place.attributes["NAME"],
                transform=ccrs.PlateCarree(),
                fontsize=10,
                ha="right",
                va="bottom",
                color="black",
                bbox=dict(facecolor="white", alpha=0.3, edgecolor="none"),
            )

    # Add gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Plot heatmap with 'Spectral' colormap centered at 0
    X, Y = np.meshgrid(lons, lats)
    max_abs_elevation = np.max(np.abs(raster))  # Symmetric range around 0
    heatmap = ax.pcolormesh(
        X,
        Y,
        raster,
        cmap="seismic",
        vmin=-max_abs_elevation,
        vmax=max_abs_elevation,
        transform=ccrs.PlateCarree(),
        alpha=0.8,
    )

    # Adjust colorbar to match map height
    cbar = plt.colorbar(heatmap, label="Elevation (m)", aspect=20, pad=0.02)
    cbar.ax.set_position(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )

    # Plot boundaries
    for i, (boundary_lons, boundary_lats) in enumerate(boundaries):
        ax.plot(
            boundary_lons,
            boundary_lats,
            color="yellow",
            linewidth=1,
            transform=ccrs.PlateCarree(),
            label="Basin outline" if i == 0 else None,
        )

    # Plot smoothing (thinner and dashed)
    if smoothing_lons and smoothing_lats:
        ax.plot(
            smoothing_lons,
            smoothing_lats,
            color="blue",
            linewidth=1,
            linestyle="--",
            transform=ccrs.PlateCarree(),
            label="Smoothing boundary",
        )

    # Set title with provided basin_name
    plt.title(f"{basin_name} Basin: Boundary and Depth of Basement", fontsize=14)

    ax.legend()

    # Save the plot to out_dir and close the figure
    output_file = out_dir / f"{basin_name}_basin_map.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Map saved to {output_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot basement heatmap, boundaries, and optional smoothing with online basemap."
    )
    parser.add_argument(
        "basin_name", type=str, help="Name of the basin for the map title"
    )
    parser.add_argument(
        "data_path", type=Path, help="Directory containing all input files"
    )
    parser.add_argument(
        "basement",
        type=Path,
        help="Path to basement file (relative to data_path if not absolute)",
    )
    parser.add_argument(
        "boundary",
        nargs="+",
        type=Path,
        help="Path(s) to boundary file(s) (relative to data_path if not absolute)",
    )
    parser.add_argument(
        "--smoothing",
        type=Path,
        default=None,
        help="Path to smoothing file (relative to data_path if not absolute)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save the output map (defaults to data_path)",
    )

    args = parser.parse_args()

    out_dir = args.data_path if args.out_dir is None else args.out_dir

    data_path = args.data_path
    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist.")
        raise ValueError

    out_dir.mkdir(parents=True, exist_ok=True)

    basement_file = (
        data_path / args.basement if not args.basement.is_absolute() else args.basement
    )
    boundary_files = [
        data_path / bf if not bf.is_absolute() else bf for bf in args.boundary
    ]
    smoothing_file = args.smoothing
    if smoothing_file:
        smoothing_file = (
            data_path / smoothing_file
            if not smoothing_file.is_absolute()
            else smoothing_file
        )

    plot_data(basement_file, boundary_files, smoothing_file, args.basin_name, out_dir)


if __name__ == "__main__":
    main()
