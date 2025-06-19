"""
Convert ASCII text file of lon-lat points to GeoJSON Polygon.

This script reads a text file containing space-separated longitude and latitude
coordinates, converts them into a GeoJSON Polygon FeatureCollection, and writes
the output to a GeoJSON file. It ensures the polygon is closed and valid. If the
polygon is not closed, it appends the closing point to the input file.

Usage:
    python boundary_txt_to_geojson.py <input_txt_file>


"""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(pretty_exceptions_enable=False)


def read_coordinates_from_file(file_path: Path) -> list[list[float]]:
    """
    Reads space-separated longitude-latitude coordinates from a text file.

    Parameters
    ----------
    file_path : Path
        Path to the input ASCII file containing lon-lat coordinates.

    Returns
    -------
    list[list[float]]
        A list of coordinates, where each coordinate is a list of [lon, lat].

    Raises
    ------
    ValueError: If the file cannot be read or contains malformed lines.

    """
    lines = file_path.read_text().splitlines()
    coordinates = []
    # Explicitly filter out empty or whitespace-only lines before processing
    for line in [ln for ln in lines if ln.strip()]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                longitude = float(parts[0])
                latitude = float(parts[1])
                coordinates.append([longitude, latitude])
            except ValueError:
                typer.echo(f"⚠️ Skipping malformed line: {line}", err=True)
    return coordinates


def create_geojson(coordinates: list[list[float]]) -> dict:
    """
    Creates a GeoJSON Polygon FeatureCollection from coordinates.

    Parameters
    ----------
    coordinates : list[list[float]]
        A list of [lon, lat] coordinate pairs.

    Returns
    -------
    dict
        A GeoJSON dictionary representing the polygon.


    """
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                "properties": {},
            }
        ],
    }


def write_geojson_to_file(geojson: dict, output_path: Path) -> None:
    """
    Writes the GeoJSON content to a file.

    Parameters
    ----------
    geojson : dict
        The GeoJSON dictionary to write.
    output_path : Path
        Path to the output GeoJSON file.

    Raises
    ------
    IOError: If the file cannot be written.

    """
    output_path.write_text(json.dumps(geojson, indent=4))
    typer.echo(f"✅ GeoJSON file written: {output_path}")


@app.command()
def convert_to_geojson(
    input_txt: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="Path to the input ASCII text file containing lon-lat coordinates",
        ),
    ],
) -> None:
    """
    Converts an ASCII text file of lon-lat points to a GeoJSON Polygon.
    Ensures the polygon is closed and valid.

    Parameters
    ----------
    input_txt : Path
        Path to the input text file containing space-separated lon-lat coordinates.


    """
    if input_txt.suffix.lower() != ".txt":
        typer.echo("❌ Input file must end with .txt", err=True)
        raise typer.Exit(code=1)

    coordinates = read_coordinates_from_file(input_txt)

    if not coordinates or len(coordinates) < 3:
        typer.echo("❌ Not enough valid coordinates to form a polygon", err=True)
        raise typer.Exit(code=1)

    if coordinates[0] != coordinates[-1]:
        typer.echo(
            "⚠️ The input is not closed — closing polygon/updating input file automatically",
            err=True,
        )
        coordinates.append(coordinates[0])  # Close the polygon in memory
        # Ensure there's a newline before appending the closing coordinate
        content = input_txt.read_text()
        with input_txt.open("a") as f:
            if content and not content.endswith("\n"):
                f.write("\n")
            f.write(f"{coordinates[-1][0]} {coordinates[-1][1]}\n")

    geojson = create_geojson(coordinates)
    output_geojson = input_txt.with_suffix(".geojson")
    write_geojson_to_file(geojson, output_geojson)


if __name__ == "__main__":
    app()
