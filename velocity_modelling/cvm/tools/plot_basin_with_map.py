import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
from pathlib import Path
import rasterio
from matplotlib.colors import LightSource
from pyproj import Transformer

#GEOLOGY_FILE = Path(__file__).parent / "NE2_HR_LC_SR_W_DR.tif"
#GEOLOGY_FILE = Path(__file__).parent / "NE2_50M_SR_W.tif"
#GEOLOGY_FILE = Path(__file__).parent / "fake.tif"
GEOLOGY_FILE = Path(__file__).parent / "NZ10.tif"

def load_basement(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: Basement file {file_path} does not exist.")
        return None, None, None
    try:
        with file_path.open('r') as f:
            lines = f.readlines()

        # Parse header
        n_lat, n_lon = map(int, lines[0].split())

        # Parse latitudes and longitudes
        lats = np.array([float(x) for x in lines[1].split()])
        lons = np.array([float(x) for x in lines[2].split()])

        # Check if number of lines matches expected
        raster_lines = lines[3:]
        if len(raster_lines) != n_lat:
            print(f"Alert: Expected {n_lat} raster lines in {file_path}, got {len(raster_lines)}")
            # Pad with zeros or truncate
            if len(raster_lines) < n_lat:
                raster_lines.extend(['0 ' * n_lon] * (n_lat - len(raster_lines)))
            raster_lines = raster_lines[:n_lat]

        # Parse raster values line by line
        raster = []
        for lat_idx, line in enumerate(raster_lines):
            values = [float(x) for x in line.split()]
            if len(values) != n_lon:
                print(f"Alert: Expected {n_lon} values in line {lat_idx + 4} of {file_path}, got {len(values)}")
                # Pad with zeros (default option 1)
                if len(values) < n_lon:
                    values.extend([0] * (n_lon - len(values)))
                # Truncate if too many values
                values = values[:n_lon]
            raster.append(values)

        # Convert to 2D numpy array
        raster = np.array(raster)

        # Update file if modifications were made
        if len(raster_lines) != n_lat or any(len(line.split()) != n_lon for line in raster_lines):
            with file_path.open('w') as f:
                f.write(f"{n_lat} {n_lon}\n")
                f.write(" ".join(map(str, lats)) + "\n")
                f.write(" ".join(map(str, lons)) + "\n")
                for row in raster:
                    f.write(" ".join(map(str, row)) + "\n")

        return lats, lons, raster

    except Exception as e:
        print(f"Error loading basement file {file_path}: {e}")
        return None, None, None


def load_boundary(file_path, is_boundary=True):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: Boundary file {file_path} does not exist.")
        return None, None
    try:
        with file_path.open('r') as f:
            lines = f.readlines()

        # Parse coordinates
        coords = [list(map(float, line.split())) for line in lines]
        lons = [coord[0] for coord in coords]
        lats = [coord[1] for coord in coords]

        # For boundary files, check if closed
        if is_boundary:
            if coords[0] != coords[-1]:
                print(f"Alert: Boundary in {file_path} is not closed. Closing it.")
                coords.append(coords[0])
                lons.append(lons[0])
                lats.append(lats[0])

                # Update file
                with file_path.open('w') as f:
                    for lon, lat in coords:
                        f.write(f"{lon} {lat}\n")

        return lons, lats

    except Exception as e:
        print(f"Error loading {'boundary' if is_boundary else 'smoothing'} file {file_path}: {e}")
        return None, None


def plot_data(basement_file : Path, boundary_files: list[Path], smoothing_file: Path, basin_name: str, out_dir: Path):


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
    lon_buffer = (lon_max - lon_min) * 0.4 or 0.5
    lat_buffer = (lat_max - lat_min) * 0.4 or 0.5
    extent = [lon_min - lon_buffer, lon_max + lon_buffer, lat_min - lat_buffer, lat_max + lat_buffer]
    print(f"Map extent: {extent}")

    # Create map
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Load and plot local geology file (NE2_50M_SR_W.tif)
    if not GEOLOGY_FILE.exists():
        print(f"Error: Geology file {GEOLOGY_FILE} not found. Falling back to grayscale LAND.")
        ax.add_feature(cfeature.LAND, facecolor='gray', alpha=0.5)
    else:
        try:
            with rasterio.open(GEOLOGY_FILE) as src:
                band_count = src.count
                print(f"Geology TIFF has {band_count} band(s)")

                for band in range(1, band_count + 1):
                    band_data = src.read(band, masked=True)
                    if np.ma.is_masked(band_data):
                        valid_band_data = band_data.compressed()
                    else:
                        valid_band_data = band_data.flatten()
                    if len(valid_band_data) > 0:
                        print(f"Band {band} range: {valid_band_data.min()} to {valid_band_data.max()}")
                    else:
                        print(f"Band {band}: No valid data")

                # Get TIFF bounds and CRS
                left, bottom, right, top = src.bounds
                tiff_crs = src.crs
                print(f"TIFF extent (original CRS): [{left}, {right}, {bottom}, {top}]")
                print(f"TIFF CRS (reported): {tiff_crs}")

                # Test 1: Try transforming assuming EPSG:2193 (NZTM2000), with swapped coordinates
                try:
                    print(
                        "Attempting transformation assuming TIFF is in EPSG:2193 (NZTM2000) with swapped coordinates...")
                    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
                    # Swap: treat [left, right, bottom, top] as [y_min, y_max, x_min, x_max]
                    lat_bottom, lat_top = transformer.transform(bottom, top)[1], transformer.transform(left, top)[1]
                    lon_left, lon_right = transformer.transform(left, bottom)[0], transformer.transform(right, bottom)[
                        0]
                    tiff_extent_wgs84 = [lon_left, lon_right, lat_bottom, lat_top]
                    print(f"TIFF extent (WGS84, from EPSG:2193, swapped): {tiff_extent_wgs84}")
                except Exception as e:
                    print(f"Transformation from EPSG:2193 with swapped coordinates failed: {e}. Trying normal order...")
                    # Test 2: Normal order with EPSG:2193
                    try:
                        transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
                        lon_left, lat_bottom = transformer.transform(left, bottom)
                        lon_right, lat_top = transformer.transform(right, top)
                        tiff_extent_wgs84 = [lon_left, lon_right, lat_bottom, lat_top]
                        print(f"TIFF extent (WGS84, from EPSG:2193, normal): {tiff_extent_wgs84}")
                    except Exception as e:
                        print(
                            f"Transformation from EPSG:2193 failed: {e}. Falling back to reported CRS ({tiff_crs})...")
                        # Test 3: Fall back to reported CRS
                        transformer = Transformer.from_crs(tiff_crs, "EPSG:4326", always_xy=True)
                        lon_left, lat_bottom = transformer.transform(left, bottom)
                        lon_right, lat_top = transformer.transform(right, top)
                        tiff_extent_wgs84 = [lon_left, lon_right, lat_bottom, lat_top]
                        print(f"TIFF extent (WGS84, from reported CRS): {tiff_extent_wgs84}")

                if band_count == 4:
                    r = src.read(1, masked=True).astype(float)
                    g = src.read(2, masked=True).astype(float)
                    b = src.read(3, masked=True).astype(float)
                    geology_data = 0.2989 * r + 0.5870 * g + 0.1140 * b

                    if np.ma.is_masked(geology_data):
                        valid_data = geology_data.compressed()
                    else:
                        valid_data = geology_data.flatten()
                    if len(valid_data) > 0:
                        min_val, max_val = valid_data.min(), valid_data.max()
                        print(f"Grayscale range: {min_val} to {max_val}")
                        if max_val > min_val:
                            normalized_geology = (geology_data - min_val) / (max_val - min_val)
                            print(f"Normalized range: {normalized_geology.min()} to {normalized_geology.max()}")
                        else:
                            normalized_geology = geology_data
                    else:
                        normalized_geology = geology_data

                    ax.imshow(normalized_geology, extent=tiff_extent_wgs84, origin='upper', cmap='gray', transform=ccrs.PlateCarree(), alpha=0.5)

                else:
                    geology_data = src.read(1, masked=True)
                    if np.ma.is_masked(geology_data):
                        valid_data = geology_data.compressed()
                    else:
                        valid_data = geology_data.flatten()
                    if len(valid_data) > 0:
                        min_val, max_val = valid_data.min(), valid_data.max()
                        print(f"Grayscale range: {min_val} to {max_val}")
                        if max_val > min_val:
                            normalized_geology = (geology_data - min_val) / (max_val - min_val)
                            print(f"Normalized range: {normalized_geology.min()} to {normalized_geology.max()}")
                        else:
                            normalized_geology = geology_data
                    else:
                        normalized_geology = geology_data

                    ax.imshow(normalized_geology, extent=tiff_extent_wgs84, origin='upper',
                              cmap='gray', transform=ccrs.PlateCarree(), alpha=0.5)

        except Exception as e:
            print(f"Error loading geology file {GEOLOGY_FILE}: {e}. Falling back to grayscale LAND.")
            ax.add_feature(cfeature.LAND, facecolor='gray', alpha=0.5)

    # Add other features
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.8)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ax.add_feature(cfeature.LAKES, edgecolor='blue', facecolor='lightblue')
    ax.add_feature(cfeature.RIVERS, edgecolor='lightblue', alpha=0.9)

    # Add town names
    shp = shpreader.natural_earth(resolution='10m', category='cultural', name='populated_places')
    reader = shpreader.Reader(shp)
    places = list(reader.records())
    for place in places:
        lon, lat = place.geometry.x, place.geometry.y
        if (extent[0] <= lon <= extent[1]) and (extent[2] <= lat <= extent[3]):
            ax.text(lon, lat, place.attributes['NAME'], transform=ccrs.PlateCarree(),
                    fontsize=8, ha='right', va='bottom', color='black',
                    bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Plot heatmap (basement) with 'hot' colormap
    X, Y = np.meshgrid(lons, lats)
    #heatmap = ax.pcolormesh(X, Y, raster, cmap='hot', transform=ccrs.PlateCarree(), alpha=0.7)

    max_abs_elevation = np.max(np.abs(raster))  # Symmetric range around 0
    heatmap = ax.pcolormesh(X, Y, raster, cmap='seismic', vmin=-max_abs_elevation, vmax=max_abs_elevation,
                            transform=ccrs.PlateCarree(), alpha=0.7)

    # Adjust colorbar to match map height
    cbar = plt.colorbar(heatmap, label='Elevation (m)', aspect=20, pad=0.02)
    cbar.ax.set_position([ax.get_position().x1 + 0.01, ax.get_position().y0,
                          0.02, ax.get_position().height])

    # Plot boundaries
    for i, (boundary_lons, boundary_lats) in enumerate(boundaries):
        ax.plot(boundary_lons, boundary_lats, color='red', linewidth=2,
                transform=ccrs.PlateCarree(), label=f'Boundary {i + 1}' if i == 0 else None)

    # Plot smoothing (thinner and dashed)
    if smoothing_lons and smoothing_lats:
        ax.plot(smoothing_lons, smoothing_lats, color='blue', linewidth=1, linestyle='--',
                transform=ccrs.PlateCarree(), label='Smoothing')

    # Set title with provided basin_name
    plt.title(f"{basin_name} Basin: Boundary and Depth of Basement", fontsize=14)

    ax.legend()

    # Save the plot to out_dir
    output_file = out_dir / f"{basin_name}_basin_map.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Map saved to {output_file}")
    plt.close(fig)
    #plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot basement heatmap, one or more boundaries, and optional smoothing on a dynamic map of New Zealand with towns and geology.")
    parser.add_argument('basin_name', type=str, help="Name of the basin for the map title")
    parser.add_argument('data_path', type=Path, help="Directory containing all input files")
    parser.add_argument('basement', type=Path,
                        help="Path to the basement file (heatmap data), relative to data_path if not absolute")
    parser.add_argument('boundary', nargs='+', type=Path,
                        help="Path(s) to the boundary file(s) (polygons, one or more), relative to data_path if not absolute")

    parser.add_argument('--smoothing', type=Path, default=None,
                        help="Path to the smoothing file (polyline, optional), relative to data_path if not absolute")
    parser.add_argument('--out-dir', type=Path, default=None,
                        help="Directory to save the output map (defaults to data_path)")

    args = parser.parse_args()

    # Set out_dir to data_path if not provided
    out_dir = args.data_path if args.out_dir is None else args.out_dir

    data_path = args.data_path
    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist.")
        raise ValueError

    out_dir.mkdir(parents=True, exist_ok=True)  # Create out_dir if it doesn't exist

    # Resolve absolute paths and check existence
    basement_file = data_path / args.basement if not args.basement.is_absolute() else args.basement
    boundary_files = [data_path / bf if not bf.is_absolute() else bf for bf in args.boundary]
    smoothing_file = args.smoothing
    if smoothing_file:
        smoothing_file = data_path / smoothing_file if not smoothing_file.is_absolute() else smoothing_file

    plot_data(basement_file, boundary_files, smoothing_file, args.basin_name, out_dir)


if __name__ == "__main__":
    main()