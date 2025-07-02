#!/usr/bin/env python
"""
view_stacked_depths.py  •  toggle vp / vs / rho  •  cropped window  •  equal layer spacing
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go


# --------------------------------------------------------------------------- #
def _is_number(txt):
    try:
        float(txt)
        return True
    except ValueError:
        return False


def load_model(h5file: Path, lat_lim=None, lon_lim=None):
    """Return (depths, lon, lat, data) where data[var] = 3-D [Nz, Ny, Nx]."""
    with h5py.File(h5file, "r") as f:
        depth_keys = sorted((k for k in f if _is_number(k)), key=float, reverse=True)
        lat_full = f[depth_keys[0]]["latitudes"][:]
        lon_full = f[depth_keys[0]]["longitudes"][:]

        lat_mask = np.ones_like(lat_full, bool)
        lon_mask = np.ones_like(lon_full, bool)
        if lat_lim:
            lat_mask &= (lat_full >= lat_lim[0]) & (lat_full <= lat_lim[1])
        if lon_lim:
            lon_mask &= (lon_full >= lon_lim[0]) & (lon_full <= lon_lim[1])

        lat = lat_full[lat_mask]
        lon = lon_full[lon_mask]
        Nz, Ny, Nx = len(depth_keys), lat.size, lon.size

        cube = {v: np.empty((Nz, Ny, Nx), dtype=np.float32) for v in ("vp", "vs", "rho")}
        for iz, dk in enumerate(depth_keys):
            g = f[dk]
            for v in cube:
                cube[v][iz] = g[v][:][np.ix_(lat_mask, lon_mask)]

    return [float(d) for d in depth_keys], lon, lat, cube

# 🔸 hovertemplate – one string reused for every surface
HOVER_TMPL = (
    "Lon: %{x:.3f}°<br>"
    "Lat: %{y:.3f}°<br>"
    "Layer idx: %{z:.3f}<br>"
    "Vp: %{customdata[0]:.2f} km/s<br>"
    "Vs: %{customdata[1]:.2f} km/s<br>"
    "Rho: %{customdata[2]:.2f} g/cm³<br>"
    "<extra></extra>"
)

def make_figure(depths, lon, lat, cube, gap=0.2, cmap="Hot_r", opacity=1.0):
    """Build Plotly figure with dropdown that switches vp/vs/rho."""
    Nz = len(depths)
    X, Y = np.meshgrid(lon, lat)
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    units = {"vp": "km/s", "vs": "km/s", "rho": "g/cm³"}
    vmin = {v: cube[v].min() for v in cube}
    vmax = {v: cube[v].max() for v in cube}

    fig, traces_by_var, t = go.Figure(), {v: [] for v in cube}, 0

    for var in cube:
        for iz in range(Nz):
            Z = np.full_like(cube[var][iz], -gap * iz)
            # 🔸 build per-cell 3-value customdata for hover
            customdata = np.stack(
                [cube['vp'][iz], cube['vs'][iz], cube['rho'][iz]], axis=-1
            )
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=cube[var][iz],
                colorscale=cmap,
                cmin=vmin[var], cmax=vmax[var],
                visible=(var == "vp"),
                showscale=False,
                opacity=opacity,
                name=f"{depths[iz]:g} km",
                legendgroup=f"lay{iz}", showlegend=True,
                customdata=customdata,
                hovertemplate=HOVER_TMPL,
                hoverinfo="skip",
            ))
            traces_by_var[var].append(t);  t += 1

        # colour-bar carrier (one per variable)
        fig.add_trace(go.Surface(
            x=[[lon_min, lon_max], [lon_min, lon_max]],
            y=[[lat_min, lat_min], [lat_max, lat_max]],
            z=[[0, 0], [0, 0]],
            surfacecolor=[[vmin[var], vmax[var]], [vmin[var], vmax[var]]],
            colorscale=cmap, cmin=vmin[var], cmax=vmax[var],
            showscale=True, showlegend=False,
            visible=(var == "vp"),
            opacity=0,
            colorbar=dict(title=f"{var.upper()} ({units[var]})")
        ))
        traces_by_var[var].append(t);  t += 1

    # dropdown menu
    buttons = []
    for var in cube:
        vis = [False]*t
        for idx in traces_by_var[var]:
            vis[idx] = True
        buttons.append(dict(label=var.upper(),
                            method="update",
                            args=[{"visible": vis}]))
    fig.update_layout(
        title="Stacked depth slices (toggle vp / vs / rho)",
        scene=dict(
            xaxis=dict(title="Longitude (°E, 0–360)"),
            yaxis=dict(title="Latitude (°N)"),
            zaxis=dict(title="Layer index (shallow = 0)"),
            aspectmode="manual",
            aspectratio=dict(x=2, y=1.2, z=0.3),
        ),
        annotations=[
            dict(
                showarrow=False,
                text="",
                x=1.13,  # Adjust position to avoid colorbar
                y=1.02,
                xref="paper",
                yref="paper",
                align="left"
            )
        ],
        updatemenus=[dict(
            buttons=buttons,
            direction="down",
            x=1.1, xanchor="left",
            y=1.0, yanchor="top",
            showactive=True,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
        )],
        legend=dict(
            title="Layers",
            x=0.0, xanchor="left",
            y=0.95, yanchor="top",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
        ),
        hoverlabel=dict(font_size=11, font_family="Arial"),
    )
    return fig


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("h5file", help="2020_NZ.h5")
    p.add_argument("--lat_range", nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument("--lon_range", nargs=2, type=float, metavar=("MIN", "MAX"))
    p.add_argument("--gap", type=float, default=0.005, help="vertical gap between layers")
    args = p.parse_args()

    depths, lon, lat, cube = load_model(
        Path(args.h5file),
        lat_lim=args.lat_range,
        lon_lim=args.lon_range
    )

    make_figure(depths, lon, lat, cube, gap=args.gap).show()

