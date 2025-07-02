#!/usr/bin/env python
"""
merge_geojson_boundaries.py
Combine any number of boundary GeoJSONs into one file and
give each source file a distinct colour.

Usage
-----
python merge_geojson_boundaries.py A.geojson B.geojson C.geojson
python merge_geojson_boundaries.py *.geojson -o all_boundaries.geojson
"""

import argparse
import itertools
import json
from pathlib import Path

import geopandas as gpd
from matplotlib import cm  # for colour palettes
from matplotlib.colors import to_hex


def colour_cycle(n):
    """
    Return an iterator that yields n visually-distinct HEX colours.
    Uses Matplotlib’s tab10 palette first, then cycles through viridis.
    """
    base = [to_hex(c) for c in cm.get_cmap("tab10").colors]
    if n <= len(base):
        return base[:n]

    # Need more colours – extend with a continuous palette
    extra_needed = n - len(base)
    extra = [to_hex(cm.viridis(i / extra_needed)) for i in range(extra_needed)]
    return base + extra


def merge_geojsons(input_paths, output_path):
    # Generate colours (repeatable order)
    colours = colour_cycle(len(input_paths))

    merged_gdfs = []

    for file_path, colour in zip(input_paths, colours):
        gdf = gpd.read_file(file_path)

        # Add styling properties for every feature in this file
        gdf["stroke"] = colour
        gdf["stroke-width"] = 1.5
        gdf["fill"] = colour
        gdf["fill-opacity"] = 0.05
        gdf["source_file"] = Path(file_path).stem  # keep provenance

        merged_gdfs.append(gdf)

    # Concatenate into a single GeoDataFrame
    merged = gpd.GeoDataFrame(
        pd.concat(merged_gdfs, ignore_index=True),
        crs=merged_gdfs[0].crs,
    )

    # Write to GeoJSON
    merged.to_file(output_path, driver="GeoJSON")
    print(f"✅  Merged file written to: {output_path}")


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Merge multiple GeoJSON boundary files into one and style each with a unique colour."
    )
    parser.add_argument(
        "geojson_files", nargs="+", help="Input GeoJSON files (wildcards allowed)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="merged_boundaries.geojson",
        help="Output filename (default: merged_boundaries.geojson)",
    )
    args = parser.parse_args()

    merge_geojsons(args.geojson_files, args.output)

