import json
import os
import random
import sys
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt


def generate_colors(n):
    cmap = plt.get_cmap("brg")  # You can choose different colormaps from matplotlib
    color_list = [cmap(random.random()) for _ in range(n)]
    colors = []
    for i in range(n):
        color = color_list[i]
        stroke = f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"
        fill = f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"
        colors.append(
            {"stroke": stroke, "fill": fill, "stroke-width": 1, "fill-opacity": 0.3}
        )
    return colors


def read_geojson(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def read_file_list(file_list_path):
    with open(file_list_path, "r") as file:
        return [Path(line.strip()) for line in file.readlines()]


def combine_geojson(files):
    combined_features = []
    groups = {}
    for b in files:
        parent = b.parent
        if parent in groups:
            groups[parent].append(b)
        else:
            groups[parent] = [b]

    print(groups)
    #    colors = generate_colors(len(groups))
    color_text = "#ba0045"
    colors = [
        {
            "stroke": color_text,
            "fill": color_text,
            "stroke-width": 1,
            "fill-opacity": 0.3,
        }
    ] * len(groups)
    color_cycle = cycle(colors)

    for parent, group in groups.items():
        color = next(color_cycle)
        print(f"{parent} {color}")
        for file_path in group:
            geojson = read_geojson(file_path)
            for feature in geojson["features"]:
                feature["properties"].update(color)
                feature["properties"]["source_file"] = os.path.basename(file_path)
                combined_features.append(feature)

    combined_geojson = {"type": "FeatureCollection", "features": combined_features}
    return combined_geojson


def write_geojson_to_file(geojson, output_file_path):
    with open(output_file_path, "w") as file:
        json.dump(geojson, file, indent=4)


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file_list> <output_file_path>")
        sys.exit(1)

    input_file_list_path = sys.argv[1]
    output_file_path = sys.argv[2]

    if not os.path.isfile(input_file_list_path):
        print(f"Error: File '{input_file_list_path}' not found.")
        sys.exit(1)

    input_files = read_file_list(input_file_list_path)

    for input_file in input_files:
        if not os.path.isfile(input_file):
            print(f"Error: File '{input_file}' not found.")
            sys.exit(1)

    combined_geojson = combine_geojson(input_files)
    write_geojson_to_file(combined_geojson, output_file_path)
    print(f"Combined GeoJSON file created: {output_file_path}")


if __name__ == "__main__":
    main()
