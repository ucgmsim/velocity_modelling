import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def load_basement(file_path):
    try:
        with open(file_path, 'r') as f:
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
            with open(file_path, 'w') as f:
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
    try:
        with open(file_path, 'r') as f:
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
                with open(f"{file_path}_fixed", 'w') as f:
                    for lon, lat in coords:
                        f.write(f"{lon} {lat}\n")

        return lons, lats

    except Exception as e:
        print(f"Error loading {'boundary' if is_boundary else 'smoothing'} file {file_path}: {e}")
        return None, None


def plot_data(basement_file, boundary_files, smoothing_file=None):
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

    # Create map
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Set extent based on basement data
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())

    # Add gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # Plot heatmap (basement)
    X, Y = np.meshgrid(lons, lats)
    heatmap = ax.pcolormesh(X, Y, raster, cmap='viridis', transform=ccrs.PlateCarree())
    plt.colorbar(heatmap, label='Raster Values')

    # Plot all boundaries
    for i, (boundary_lons, boundary_lats) in enumerate(boundaries):
        ax.plot(boundary_lons, boundary_lats, color='red', linewidth=2,
                transform=ccrs.PlateCarree(), label=f'Boundary {i + 1}' if i == 0 else None)

    # Plot smoothing if provided
    if smoothing_lons and smoothing_lats:
        ax.plot(smoothing_lons, smoothing_lats, color='blue', linewidth=2,
                transform=ccrs.PlateCarree(), label='Smoothing')

    ax.legend()
    ax.coastlines()
    plt.title('Map with Heatmap, Boundaries, and Optional Smoothing')
    plt.show()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Plot basement heatmap, one or more boundaries, and optional smoothing on a map.")
    parser.add_argument('basement', type=str, help="Path to the basement file (heatmap data)")
    parser.add_argument('boundary', nargs='+', type=str,
                        help="Path(s) to the boundary file(s) (polygons, one or more)")
    parser.add_argument('--smoothing', type=str, default=None,
                        help="Path to the smoothing file (polyline, optional)")

    # Parse arguments
    args = parser.parse_args()

    # Call plotting function with provided files
    plot_data(args.basement, args.boundary, args.smoothing)


if __name__ == "__main__":
    main()