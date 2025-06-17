from typing import Annotated
from pathlib import Path
import typer
import json

app = typer.Typer(pretty_exceptions_enable=False)


def read_coordinates_from_file(file_path: Path) -> list[list[float]]:
    """
    Reads space-separated longitude-latitude coordinates from a text file.

    Args:
        file_path: Path to the input ASCII file.

    Returns:
        List of [lon, lat] coordinate pairs.
    """
    lines = file_path.read_text().splitlines()
    coordinates = []
    for line in lines:
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

    Args:
        coordinates: List of [lon, lat] coordinate pairs.

    Returns:
        A GeoJSON dictionary.
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

    Args:
        geojson: The GeoJSON dictionary to write.
        output_path: Path to the output GeoJSON file.
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
    ]
) -> None:
    """
    Converts an ASCII text file of lon-lat points to a GeoJSON Polygon.
    Ensures the polygon is closed and valid.
    """
    if input_txt.suffix.lower() != ".txt":
        typer.echo("❌ Input file must end with .txt", err=True)
        raise typer.Exit(code=1)

    coordinates = read_coordinates_from_file(input_txt)

    if not coordinates or len(coordinates) < 3:
        typer.echo("❌ Not enough valid coordinates to form a polygon", err=True)
        raise typer.Exit(code=1)

    if coordinates[0] != coordinates[-1]:
        typer.echo("⚠️ The input is not closed — closing polygon/updating input file automatically", err=True)
        coordinates.append(coordinates[0])  # Close the polygon in memory
        with input_txt.open("a") as f:
            f.write(f"{coordinates[-1][0]} {coordinates[-1][1]}\n")
            


    geojson = create_geojson(coordinates)
    output_geojson = input_txt.with_suffix(".geojson")
    write_geojson_to_file(geojson, output_geojson)


if __name__ == "__main__":
    app()

