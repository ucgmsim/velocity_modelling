import json
from pathlib import Path
from itertools import cycle
import typer
import matplotlib.pyplot as plt

def generate_colors(n):
    """
    Generates a list of colors.

    Parameters
    ----------
    n : int
        The number of colors to generate.

    Returns
    -------
    list
        A list of color dictionaries with 'stroke', 'fill', and 'stroke-width'.
    """
    cmap = plt.get_cmap('tab20')
    colors = []
    for i in range(n):
        color = cmap(i % 20)
        stroke = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        fill = '#{:02x}{:02x}{:02x}'.format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        colors.append({"stroke": stroke, "fill": fill, "stroke-width": 1, "fill-opacity": 0.5})
    return colors

def read_geojson(file_path: Path):
    """
    Reads a GeoJSON file.

    Parameters
    ----------
    file_path : Path
        The path to the input GeoJSON file.

    Returns
    -------
    dict
        The content of the GeoJSON file.
    """
    with file_path.open('r') as file:
        return json.load(file)

def read_file_list(file_list_path: Path):
    """
    Reads a list of file paths from a text file.

    Parameters
    ----------
    file_list_path : Path
        The path to the file containing the list of GeoJSON files.

    Returns
    -------
    list
        A list of file paths as strings.
    """
    with file_list_path.open('r') as file:
        return [line.strip() for line in file.readlines()]

def combine_geojson(files):
    """
    Combines multiple GeoJSON files into one.

    Parameters
    ----------
    files : list
        A list of file paths to GeoJSON files.

    Returns
    -------
    dict
        A combined GeoJSON object.
    """
    combined_features = []
    colors = generate_colors(len(files))
    color_cycle = cycle(colors)

    for file_path in files:
        geojson = read_geojson(Path(file_path))
        color = next(color_cycle)
        for feature in geojson['features']:
            feature['properties'].update(color)
            feature['properties']['source_file'] = Path(file_path).name
            combined_features.append(feature)

    combined_geojson = {
        "type": "FeatureCollection",
        "features": combined_features
    }
    return combined_geojson

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

def main(input_file_list: Path, output_file: Path):
    """
    Combines multiple GeoJSON files into one and assigns different colors.

    Parameters
    ----------
    input_file_list : Path
        The path to the file containing the list of input GeoJSON files.
    output_file : Path
        The path to the output GeoJSON file.
    """
    input_files = read_file_list(input_file_list)

    for input_file in input_files:
        if not Path(input_file).is_file():
            print(f"Error: File '{input_file}' not found.")
            sys.exit(1)

    combined_geojson = combine_geojson(input_files)
    write_geojson_to_file(combined_geojson, output_file)
    print(f"Combined GeoJSON file created: {output_file}")

if __name__ == "__main__":
    typer.run(main)

