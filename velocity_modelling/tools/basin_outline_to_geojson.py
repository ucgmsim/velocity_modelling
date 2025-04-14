import json
import os
import sys


def read_coordinates_from_file(file_path):
    with open(file_path, "r") as file:
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
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coordinates]},
                "properties": {},
            }
        ],
    }
    return geojson


def write_geojson_to_file(geojson, output_file_path):
    with open(output_file_path, "w") as file:
        json.dump(geojson, file, indent=4)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    if not os.path.isfile(input_file_path):
        print(f"Error: File '{input_file_path}' not found.")
        sys.exit(1)

    output_file_path = os.path.splitext(input_file_path)[0] + ".geojson"

    coordinates = read_coordinates_from_file(input_file_path)
    geojson = create_geojson(coordinates)
    write_geojson_to_file(geojson, output_file_path)
    print(f"GeoJSON file created: {output_file_path}")


if __name__ == "__main__":
    main()
