#!/usr/bin/env python
"""
Merges multiple GeoJSON boundary files into a single file.

This script combines multiple GeoJSON files, assigning a unique and visually
distinct style (color, stroke) to the features from each source file.
This is useful for visualizing and comparing different geographical boundaries
on a single map. The script preserves the source file name for each feature
in the merged output for provenance.
"""

from pathlib import Path
from typing import Annotated

import geopandas as gpd
import pandas as pd
import typer
from fiona.errors import DriverError
from matplotlib import cm  # for colour palettes
from matplotlib.colors import to_hex

from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


def colour_cycle(n: int) -> list[str]:
    """Return an iterator that yields n visually-distinct HEX colours.

    Uses Matplotlib’s tab10 palette first, then cycles through viridis
    if more colours are needed.

    Parameters
    ----------
    n : int
        The number of distinct colours to generate.

    Returns
    -------
    list[str]
        A list of n HEX colour strings.
    """
    base = [to_hex(c) for c in cm.get_cmap("tab10").colors]
    if n <= len(base):
        return base[:n]

    # Need more colours – extend with a continuous palette
    extra_needed = n - len(base)
    extra = [to_hex(cm.viridis(i / extra_needed)) for i in range(extra_needed)]
    return base + extra


@cli.from_docstring(app)
def merge_geojson_boundaries(
    input_paths: Annotated[
        list[Path],
        typer.Argument(
            help="One or more input GeoJSON files to merge (wildcards supported).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path for the merged output GeoJSON file.",
        ),
    ] = Path("merged_boundaries.geojson"),
) -> None:
    """Merge multiple GeoJSON boundary files into one and style each with a unique colour.

    Parameters
    ----------
    input_paths : list[Path]
        List of input GeoJSON files to merge. Supports wildcards.
    output_path : Path
        Path for the output merged GeoJSON file. Defaults to "merged_boundaries.geojson".

    Raises
    ------
    FileNotFoundError
        If any of the input files do not exist. Possibly unnecessary if using `typer.Argument` with `exists=True`.
    IOError
        If the output file cannot be written.
    ValueError
        If the input files do not contain valid geometries.
    """
    if not input_paths:
        print("No input files provided. Exiting.")
        return

    # Generate colours (repeatable order)
    colours = colour_cycle(len(input_paths))

    merged_gdfs = []

    for file_path, colour in zip(input_paths, colours):
        try:
            gdf = gpd.read_file(file_path)
            if gdf.empty:
                print(f"⚠️  Warning: {file_path.name} is empty. Skipping.")
                continue
        except (FileNotFoundError, IOError, DriverError, ValueError) as e:
            print(f"❌  Error reading {file_path.name}: {e}. Skipping.")
            continue

        # Add styling properties for every feature in this file
        gdf["stroke"] = colour
        gdf["stroke-width"] = 1.5
        gdf["fill"] = colour
        gdf["fill-opacity"] = 0.05
        gdf["source_file"] = file_path.stem  # keep provenance

        merged_gdfs.append(gdf)

    if not merged_gdfs:
        print("No valid GeoJSON files could be read. No output file created.")
        return

    # Concatenate into a single GeoDataFrame
    try:
        merged = gpd.GeoDataFrame(
            pd.concat(merged_gdfs, ignore_index=True),
            crs=merged_gdfs[0].crs,
        )

        # Write to GeoJSON
        merged.to_file(output_path, driver="GeoJSON")
        print(f"✅  Merged file written to: {output_path}")
    except (ValueError, TypeError) as e:
        print(f"❌  Failed to merge GeoDataFrames due to data incompatibility: {e}")
    except IOError as e:
        print(f"❌  Failed to write merged file to {output_path}: {e}")


if __name__ == "__main__":
    app()
