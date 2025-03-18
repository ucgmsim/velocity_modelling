"""
Generate Markdown files for basins from nzvm_registry.yaml

This script reads the nzvm_registry.yaml file and generates a Markdown file for each unique basin name.
The Markdown file contains details such as the basin type, author, images, notes, boundaries, surfaces, and smoothing boundaries. The generated files are saved in the wiki/basins directory.

Usage:
    python generate_basin_wiki.py nzvm_registry.yaml

Output .md5 files are saved in the wiki/basins directory.

"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import pytz
import yaml

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Generate Markdown files for basins from nzvm_registry.yaml"
)
parser.add_argument("yaml_file", type=Path, help="Path to the nzvm_registry.yaml file")
args = parser.parse_args()

# Get the YAML file path from arguments
yaml_file_path = args.yaml_file
if not yaml_file_path.is_file():
    print(f"Error: {yaml_file_path} does not exist or is not a file")
    exit(1)

# Define the project root directory (four levels up from script location)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "wiki" / "basins"

# Ensure the output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read the YAML content
with yaml_file_path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

# Extract basin entries
basins = data.get("basin", [])

# Dictionary to store basins by name and their versions
basin_versions = {}

# Process each basin to group by name and version
for basin in basins:
    full_name = basin.get("name", "Unnamed Basin")
    # Split at the last "_v" followed by version (e.g., "_v18p1")
    match = re.match(r"^(.*)_v(\d+p\d+)$", full_name)
    if match:
        basin_name, version = match.groups()
        # Convert version to a tuple of integers for comparison (e.g., "v18p1" -> (18, 1))
        version_parts = version.replace("v", "").split("p")
        version_tuple = (int(version_parts[0]), int(version_parts[1]))
    else:
        basin_name, version = full_name, "N/A"
        version_tuple = (0, 0)  # Default for non-versioned names

    if basin_name not in basin_versions:
        basin_versions[basin_name] = []
    basin_versions[basin_name].append(
        {
            "full_name": full_name,
            "version": version,
            "version_tuple": version_tuple,
            "data": basin,
        }
    )

# Process only the latest version of each basin
for basin_name, versions in basin_versions.items():
    # Sort versions by version_tuple to find the latest
    latest_version = max(versions, key=lambda x: x["version_tuple"])
    older_versions = [v for v in versions if v != latest_version]

    # Extract details from the latest version
    full_name = latest_version["full_name"]
    version = latest_version["version"]
    basin_data = latest_version["data"]

    basin_type = basin_data.get("type", "N/A")
    author = basin_data.get("author", "Unknown")
    images = basin_data.get("images", [])
    notes = basin_data.get("notes", [])
    boundaries = basin_data.get("boundaries", [])
    surfaces = basin_data.get("surfaces", [])
    smoothing = basin_data.get("smoothing", "N/A")

    # Derive "Created" from the latest version (e.g., v20p7 -> 2020-07)
    created = "Unknown"
    if "p" in version:
        year, month = version.split("p")
        year = f"20{year}"
        month = f"{int(month):02d}"  # Ensure two-digit month (e.g., "7" -> "07")
        created = f"{year}-{month}"

    # Construct the Markdown content
    md_content = f"# Basin : {basin_name}\n\n"

    # Overview Section
    md_content += "## Overview\n"
    md_content += "|         |                     |\n"
    md_content += "|---------|---------------------|\n"
    md_content += f"| Version | {version}           |\n"
    md_content += f"| Type    | {basin_type}        |\n"
    md_content += f"| Author  | {author}            |\n"
    md_content += f"| Created | {created}           |\n"
    if older_versions:
        md_content += (
            "| Older Versions | "
            + ", ".join(v["version"] for v in older_versions)
            + " |\n"
        )
    md_content += "\n\n"

    # Images Section
    if images:
        md_content += "## Images\n"
        image_descriptions = (
            ["Location", f"{basin_name} Basement"] if len(images) >= 2 else ["Image 1"]
        )
        for i, img in enumerate(images):
            updated_img_path = f"../images/{img}"
            description = (
                image_descriptions[i]
                if i < len(image_descriptions)
                else f"Image {i + 1}"
            )
            md_content += f"![]({updated_img_path})\n*Figure {i+1} {description}*\n"
        md_content += "\n"

    # Notes Section
    if notes:
        md_content += "## Notes\n"
        for note in notes:
            md_content += f"- {note}\n"
        md_content += "\n"

    # Data Section
    md_content += "## Data\n"

    # Boundaries Subsection
    if boundaries:
        md_content += "### Boundaries\n"
        for boundary in boundaries:
            filename = Path(boundary).name
            updated_boundary_path = f"../../velocity_modelling/Data/{boundary}"
            md_content += f"- [{filename}]({updated_boundary_path})\n"
        md_content += "\n"

    # Surfaces Subsection
    if surfaces:
        md_content += "### Surfaces\n"
        for surface in surfaces:
            surface_name = surface.get("name", "Unnamed Surface")
            submodel = surface.get("submodel", "N/A")
            surface_entry = next(
                (s for s in data["surface"] if s["name"] == surface_name), None
            )
            surface_path = surface_entry["path"] if surface_entry else "Path not found"
            updated_surface_path = f"../../velocity_modelling/Data/{surface_path}"
            md_content += (
                f"- [{surface_name}]({updated_surface_path}) (Submodel: {submodel})\n"
            )
        md_content += "\n"

    # Smoothing Boundaries Subsection
    if smoothing != "N/A":
        md_content += "### Smoothing Boundaries\n"
        smoothing_filename = Path(smoothing).name
        updated_smoothing_path = f"../../velocity_modelling/Data/{smoothing}"
        md_content += f"- [{smoothing_filename}]({updated_smoothing_path})\n"
        md_content += "\n"

    # Add timestamp at the bottom in NZ time
    nz_tz = pytz.timezone("Pacific/Auckland")
    timestamp = datetime.now(nz_tz).strftime("%B %d, %Y, %H:%M NZST/NZDT")
    md_content += f"---\n*Page generated on: {timestamp}*\n"

    # Write to a .md file using only the basin_name
    filename = OUTPUT_DIR / f"{basin_name}.md"
    with filename.open("w", encoding="utf-8") as f:
        f.write(md_content)

# Print the number of files generated (number of unique basin names)
print(f"Generated {len(basin_versions)} .md files in {OUTPUT_DIR}")
