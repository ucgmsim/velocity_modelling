from typing import Annotated
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import hsv_to_rgb
import numpy as np
from collections import defaultdict
import typer
import os

os.environ["OGR_GEOMETRY_ACCEPT_UNCLOSED_RING"] = "NO"

app = typer.Typer(pretty_exceptions_enable=False)

# Generate N distinct colors
def generate_distinct_colors(n: int):
    hues = np.linspace(0, 1, n, endpoint=False)
    colors = hsv_to_rgb(np.stack([hues, np.ones(n)*0.5, np.ones(n)*0.85], axis=1))
    return colors

def assign_region(geojson_path: Path, regions_gdf: gpd.GeoDataFrame) -> str:
    try:
        basin = gpd.read_file(geojson_path).to_crs("EPSG:4326")
        if basin.empty:
            return "Uncategorized"
        # Get the centroid of the basin polygon(s)
        centroid = basin.geometry.union_all().centroid
        matched = regions_gdf[regions_gdf.contains(centroid)]
        if not matched.empty:
            return matched.iloc[0]["REGC2025_1"]  # Adjust field name if needed
        else:
            return "Uncategorized"
    except Exception as e:
        typer.echo(f"⚠️ Failed to assign region for {geojson_path.name}: {e}", err=True)
        return "Uncategorized"

@app.command()
def generate_basin_map(
    geojson_list_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Path to a text file listing GeoJSON boundary files (one per line)",
        ),
    ],
    out_png: Annotated[
        Path,
        typer.Argument(
            dir_okay=False,
            help="Path to the output PNG image file",
        ),
    ],
    region_shapefile: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Path to StatsNZ regional council shapefile (.shp)",
        ),
    ]
) -> None:
    """
    Generate a map of all basin boundaries and classify each basin into a region
    using the provided StatsNZ regional shapefile. Outputs a PNG and a TXT file.
    """

    regions = gpd.read_file(region_shapefile).to_crs("EPSG:4326")

    basin_map: dict[str, list[Path]] = defaultdict(list)
    region_results: dict[str, str] = {}  # basin -> region

    for line in geojson_list_file.read_text().splitlines():
        line = line.strip()
        if not line or not line.endswith(".geojson"):
            continue
        path = Path(line)
        basin_name = path.parent.name if path.parent.name else "Unknown"
        basin_map[basin_name].append(path)

    num_basins = len(basin_map)
    colors = generate_distinct_colors(num_basins)

    fig, ax = plt.subplots(figsize=(18, 20))
    ax.set_xlim(165, 180)
    ax.set_ylim(-48.5, -33)

    # Add NZ basemap
    nz_basemap_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    try:
        world = gpd.read_file(nz_basemap_url)
        nz = world[world["name"] == "New Zealand"]
        nz.to_crs("EPSG:4326").plot(ax=ax, color="lightgrey", edgecolor="black")
    except Exception as e:
        typer.echo(f"⚠️ Could not load New Zealand basemap: {e}", err=True)

    legend_handles = []
    for i, (basin_name, geojson_files) in enumerate(sorted(basin_map.items())):
        color = colors[i]
        any_valid = False

        # Assign region using first valid file
        region_assigned = "Uncategorized"
        for geojson_path in geojson_files:
            try:
                region_assigned = assign_region(geojson_path, regions)
                break  # only use first valid
            except:
                continue

        region_results[basin_name] = region_assigned

        for geojson_path in geojson_files:
            try:
                gdf = gpd.read_file(geojson_path)
                gdf["geometry"] = gdf["geometry"].buffer(0)
                gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
                if gdf.empty:
                    continue
                gdf = gdf.to_crs("EPSG:4326")
                gdf.plot(ax=ax, color=color, edgecolor="black", linewidth=0.5)
                any_valid = True
            except Exception as e:
                typer.echo(f"⚠️ Error reading {geojson_path}: {e}", err=True)

        if any_valid:
            legend_handles.append(mpatches.Patch(color=color, label=basin_name))

    ax.set_title("NZ Basin Boundaries", fontsize=16)
    ax.axis("off")
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    typer.echo(f"✅ Map saved to {out_png}")

    # Save region results to .txt
    out_txt = out_png.with_suffix(".txt")
    with open(out_txt, "w") as f:
        for basin, region in sorted(region_results.items()):
            f.write(f"{basin}, {region}\n")

    typer.echo(f"✅ Region assignments written to {out_txt}")

if __name__ == "__main__":
    app()
