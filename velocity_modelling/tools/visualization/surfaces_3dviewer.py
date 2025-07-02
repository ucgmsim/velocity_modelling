import argparse
import h5py
import numpy as np
import plotly.graph_objects as go
import requests
from io import BytesIO
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

def resample_image_to_grid(img, lat_img, lon_img, lat_grid, lon_grid):
    """Resample RGB image to DEM grid coordinates."""
    interp = lambda ch: RegularGridInterpolator((lat_img, lon_img), img[:, :, ch])
    pts = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    resampled = np.stack([interp(c)(pts).reshape(lat_grid.shape) for c in range(3)], axis=-1)
    return resampled

def fetch_esri_export(lat_min, lat_max, lon_min, lon_max, px=2048, py=2048):
    """Fetch Esri World Imagery covering the bounding box."""
    url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "bboxSR": 4326, "size": f"{px},{py}", "imageSR": 4326,
        "format": "png", "f": "image"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return np.asarray(Image.open(BytesIO(r.content)).convert("RGB"), dtype=np.uint8) / 255.0

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

def plot_stacked_surfaces(data_dict, basemap_surface=None, crop_bounds=None, elev_range=None):
    fig = go.Figure()
    vmax = elev_range[-1]
    vmin = -vmax


    for idx, (file_name, (lat, lon, elevation)) in enumerate(data_dict.items()):
        if idx == 0 and basemap_surface is not None:
            # Crop to lower-layer extent
            if crop_bounds:
                lat_min_c, lat_max_c, lon_min_c, lon_max_c = crop_bounds
                lat_mask = (lat >= lat_min_c) & (lat <= lat_max_c)
                lon_mask = (lon >= lon_min_c) & (lon <= lon_max_c)
                lat = lat[lat_mask]
                lon = lon[lon_mask]
                elevation = elevation[np.ix_(lat_mask, lon_mask)]
                basemap_surface = basemap_surface[np.ix_(lat_mask, lon_mask)]
            # Convert RGB to grayscale luminance
            luminance = (
                0.2989 * basemap_surface[:, :, 0] +
                0.5870 * basemap_surface[:, :, 1] +
                0.1140 * basemap_surface[:, :, 2]
            )
            fig.add_trace(go.Surface(
                z=elevation,
                x=np.meshgrid(lon, lat)[0],
                y=np.meshgrid(lon, lat)[1],
                surfacecolor=luminance,
                colorscale="gray",
                cmin=0, cmax=1,
                showscale=False,
                name=file_name,
                legendgroup=file_name,
                opacity=1.0,
                showlegend=True,
            ))
        else:
            fig.add_trace(go.Surface(
                z=elevation,
                x=np.meshgrid(lon, lat)[0],
                y=np.meshgrid(lon, lat)[1],
                surfacecolor=elevation,
                colorscale="RdBu",
                cmin=vmin, cmax=vmax,
                showscale=False,
                opacity=0.5,
                name=file_name,
                legendgroup=file_name,
                showlegend=True,
            ))

    if crop_bounds:
        lat_min, lat_max, lon_min, lon_max = crop_bounds
        lat_range, lon_range = [lat_min, lat_max], [lon_min, lon_max]
    else:
        lat_range = lon_range = [0, 1]  # fallback if needed

    fig.update_layout(
        title="Stacked 3D Terrain Surfaces",
        scene=dict(
            xaxis=dict(title="Longitude", range=lon_range),
            yaxis=dict(title="Latitude", range=lat_range),
            zaxis=dict(title="Elevation", range=elev_range),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.1)
        ),
        legend=dict(title="Toggle Datasets", itemsizing="constant"),
        hoverlabel=dict(font_size=12, font_family="Arial", font_color="black", namelength=-1),
    )
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize stacked terrain surfaces with transparency toggles and color legend.")
    parser.add_argument("files", nargs="+", help="List of HDF5 files to process.")
    parser.add_argument("--lat_range", nargs=2, type=float, default=[-45, -42])
    parser.add_argument("--lon_range", nargs=2, type=float, default=[170, 174])
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
    img = fetch_esri_export(lat_min, lat_max, lon_min, lon_max)
    basemap_rgb = resample_image_to_grid(
        img,
        np.linspace(lat_min, lat_max, img.shape[0]),
        np.linspace(lon_min, lon_max, img.shape[1]),
        *np.meshgrid(lon0, lat0)
    )

    # Crop bounds from lower layers
    lower_bounds = list(data_dict.values())[1:]
    lower_lat = np.concatenate([v[0] for v in lower_bounds])
    lower_lon = np.concatenate([v[1] for v in lower_bounds])
    crop_bounds = (lower_lat.min(), lower_lat.max(), lower_lon.min(), lower_lon.max())

    plot_stacked_surfaces(
        data_dict,
        basemap_surface=basemap_rgb,
        crop_bounds=crop_bounds,
        elev_range=elev_range
    )

