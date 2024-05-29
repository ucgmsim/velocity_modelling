import json
from pathlib import Path
import typer

def read_coordinates_from_file(file_path: Path):
    """
    Reads coordinates from a text file.

    Parameters
    ----------
    file_path : Path
        The path to the input text file.

    Returns
    -------
    list
        A list of coordinates as [longitude, latitude].
    """
    with file_path.open('r') as file:
        lines = file.readlines()
    coordinates = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            longitude = float(parts[0])
            latitude = float(parts[1])
            coordinates.append([longitude, latitude])
    return coordinates

def create_geojson(coordinates):
    """
    Creates a GeoJSON object from coordinates.

    Parameters
    ----------
    coordinates : list
        A list of coordinates as [longitude, latitude].

    Returns
    -------
    dict
        A GeoJSON object.
    """
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates]
                },
                "properties": {}
            }
        ]
    }
    return geojson

def write_geojson_to_file(geojson, output_file_path: Path):
    """
    Writes a GeoJSON object to a file.

    Parameters
    ----------
    geojson : dict
        A GeoJSON object.
    output_file_path : Path
        The path to the output GeoJSON file.
    """
    with output_file_path.open('w') as file:
        json.dump(geojson, file, indent=4)

def main(input_file: Path):
    """
    Converts a text file of coordinates into a GeoJSON file.

    Parameters
    ----------
    input_file : Path
        The path to the input text file.
    """
    output_file_path = input_file.with_suffix('.geojson')

    coordinates = read_coordinates_from_file(input_file)
    geojson = create_geojson(coordinates)
    write_geojson_to_file(geojson, output_file_path)
    print(f"GeoJSON file created: {output_file_path}")

if __name__ == "__main__":
    typer.run(main)

