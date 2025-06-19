"""
Generate Markdown files for basins from nzcvm_registry.yaml and provide related utilities.

This script reads the nzcvm_registry.yaml file to perform operations related to basins,
such as generating wiki pages or listing available basins.

Usage examples:
  # List all available basins
  python basin_wiki.py list-basins

  # Generate wiki page for a single basin
  python basin_wiki.py generate-wiki Canterbury

  # Generate wiki pages for all basins
  python basin_wiki.py generate-wiki all

  # Generate wiki pages for all basins and scale images
  python basin_wiki.py generate-wiki all --scale-images

  # Use a specific registry file
  python basin_wiki.py list-basins --registry /path/to/my_registry.yaml
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pytz
import typer
import yaml

from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


def _get_basin_versions(registry_path: Path):
    """
    Reads and parses the registry yaml file to group models by basin name.

    Parameters
    ----------
    registry_path : Path
        Path to the nzcvm_registry.yaml file.

    Returns
    -------
    dict
        A dictionary where keys are basin names and values are lists of dictionaries
        containing basin details, including full name, version, version tuple, and data.

    Raises
    -------
    FileNotFoundError
        If the registry file does not exist at the specified path.
    ValueError
        If the registry file is not in the expected format or does not contain basin data.

    Notes
    -----
    The function expects the registry file to contain a "basin" key with a list of basin entries.
    Each basin entry should have a "name" field formatted as "BasinName_vXvY" where X and Y
    are integers representing the version.
    The version is split into a tuple of integers (major, minor) for easy comparison.

    Example
    -------
    >>> registry_path = Path("nzcvm_registry.yaml")
    >>> basin_versions = _get_basin_versions(registry_path)
    >>> print(basin_versions["Canterbury"])
    [
        {
            "full_name": "Canterbury_v20p7",
            "version": "v20p7",
            "version_tuple": (20, 7),
            "data": {...}
        },
        ...
    ]



    """

    if not registry_path.exists():
        print(f"Error: Registry file not found at {registry_path}")
        raise typer.Exit(code=1)

    with open(registry_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    basin_versions = {}
    # The basin data can be a list of single-item dicts
    for basin_item in data.get("basin", []):
        full_name = basin_item.pop("name")
        match = re.match(r"^(.*?)_v(\d+p\d+)$", full_name)
        if not match:
            continue
        basin_name, version = match.groups()
        version_parts = version.replace("v", "").split("p")
        version_tuple = (int(version_parts[0]), int(version_parts[1]))

        if basin_name not in basin_versions:
            basin_versions[basin_name] = []
        basin_versions[basin_name].append(
            {
                "full_name": full_name,
                "version": version,
                "version_tuple": version_tuple,
                "data": basin_item,
            }
        )

    return basin_versions


@app.command()
def list_basins(
    registry: Annotated[
        Path,
        typer.Option(
            "--registry",
            help="Path to the nzcvm_registry.yaml file (default: ../nzcvm_registry.yaml)",
            default_factory=lambda: Path(__file__).parent.parent
            / "nzcvm_registry.yaml",
        ),
    ],
):
    """
    Lists all unique basin names found in the registry.

    Parameters
    ----------
    registry : Path
        Path to the nzcvm_registry.yaml file. Defaults to ../nzcvm_registry.yaml.

    """
    basin_versions = _get_basin_versions(registry)
    for basin_name in sorted(basin_versions.keys()):
        print(basin_name)


@cli.from_docstring(app)
def generate_wiki(
    basin: Annotated[
        str,
        typer.Argument(
            help="Basin name to generate wiki for, or 'all' for all basins.",
        ),
    ],
    registry: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Path to the nzcvm_registry.yaml file",
        ),
    ] = Path(__file__).parent.parent / "nzcvm_registry.yaml",
    scale_images: Annotated[
        bool,
        typer.Option(
            help="Scale images to 75% size with a clickable link to full size",
        ),
    ] = False,
) -> None:
    """
    Generate Markdown files for basins from nzcvm_registry.yaml

    This command reads the nzcvm_registry.yaml file and generates a Markdown file
    for each unique basin name. The Markdown file contains details such as the basin type,
    author, images, notes, boundaries, surfaces, and smoothing boundaries.

    Parameters
    ----------
    basin : str
        Basin name to generate wiki for, or 'all' for all basins.
    registry : Path, optional
        Path to the nzcvm_registry.yaml file
    scale_images : bool, optional
        If True, scale images to 75% size with a clickable link to full size

    """
    basin_versions = _get_basin_versions(registry)

    # If a specific basin is requested, filter basin_versions
    if basin != "all":
        if basin not in basin_versions:
            print(f"Error: Basin '{basin}' not found in registry.")
            print(f"Available basins: {', '.join(sorted(basin_versions.keys()))}")
            raise typer.Exit(code=1)
        basin_versions = {basin: basin_versions[basin]}

    # Define the project root directory (four levels up from script location)
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "wiki" / "basins"

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process only the latest version of each basin
    for basin_name, versions in basin_versions.items():
        # Sort versions by version_tuple to find the latest
        latest_version = max(versions, key=lambda x: x["version_tuple"])
        older_versions = [v for v in versions if v != latest_version]

        # Extract details from the latest version
        version = latest_version["version"]
        basin_data = latest_version["data"]

        basin_type = basin_data.get("type", "N/A")
        author = basin_data.get("author", "Unknown")
        images = basin_data.get("wiki_images", [])
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

            for i, img in enumerate(images):
                description = (
                    "Location"
                    if i == 0
                    else " ".join([s.capitalize() for s in Path(img).stem.split("_")])
                )  # Use image filename as description

                updated_img_path = f"../images/{img}"

                if scale_images:
                    md_content += f'<a href="{updated_img_path}"><img src="{updated_img_path}" width="75%"></a>\n\n*Figure {i + 1} {description}*\n\n'
                else:
                    md_content += (
                        f"![]({updated_img_path})\n\n*Figure {i + 1} {description}*\n\n"
                    )
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
                file_path = Path(boundary)
                base_path = file_path.parent / file_path.stem

                # Check if alternative formats exist
                geojson_path = f"{base_path}.geojson"
                txt_path = f"{base_path}.txt"

                # Use CVM_DATA path instead of GitHub URL
                updated_txt_path = f"../../velocity_modelling/data/{txt_path}"
                updated_geojson_path = f"../../velocity_modelling/data/{geojson_path}"

                # Create links with format indicators
                links = []
                if (output_dir / updated_txt_path).exists():
                    links.append(f"[TXT]({updated_txt_path})")
                if (output_dir / updated_geojson_path).exists():
                    links.append(f"[GeoJSON]({updated_geojson_path})")

                link_text = " / ".join(links)
                md_content += f"- {file_path.stem} : {link_text}\n"
            md_content += "\n"

        # Surfaces Subsection
        if surfaces:
            md_content += "### Surfaces\n"
            for surface in surfaces:
                # Handle direct path to surface files
                surface_path = surface.get("path", "Path not found")
                file_path = Path(surface_path)
                base_path = file_path.parent / file_path.stem
                submodel = surface.get("submodel", "N/A")

                # Check if alternative formats exist
                h5_path = f"{base_path}.h5"
                in_path = f"{base_path}.in"

                # Use CVM_DATA path instead of GitHub URL
                updated_h5_path = f"../../velocity_modelling/data/{h5_path}"
                updated_in_path = f"../../velocity_modelling/data/{in_path}"

                # Create links with format indicators
                links = []
                if (output_dir / updated_h5_path).exists():
                    links.append(f"[HDF5]({updated_h5_path})")
                if (output_dir / updated_in_path).exists():
                    links.append(f"[TXT]({updated_in_path})")

                link_text = " / ".join(links)
                md_content += (
                    f"- {file_path.stem} : {link_text} (Submodel: {submodel})\n"
                )
            md_content += "\n"

        # Smoothing Boundaries Subsection
        if smoothing != "N/A":
            md_content += "### Smoothing Boundaries\n"
            smoothing_filename = Path(smoothing).name
            # Use CVM_DATA path instead of GitHub URL
            updated_smoothing_path = f"../../velocity_modelling/data/{smoothing}"
            md_content += f"- [{smoothing_filename}]({updated_smoothing_path})\n"
            md_content += "\n"

        # Retrieved From Subsection
        retrieved_from = basin_data.get("retrieved_from", {})
        if retrieved_from:
            md_content += "## Data retrieved from\n"
            for item in retrieved_from:
                for category, paths in item.items():
                    md_content += f"### {category.capitalize()}\n"
                    for path in paths:
                        filename = Path(path).name
                        # Use GitHub URL for retrieved_from paths
                        updated_path = f"https://github.com/ucgmsim/Velocity-Model/tree/main/Data/{path}"
                        md_content += f"- [{filename}]({updated_path})\n"
                    md_content += "\n"

        # Add timestamp at the bottom in NZ time
        nz_tz = pytz.timezone("Pacific/Auckland")
        timestamp = datetime.now(nz_tz).strftime("%B %d, %Y, %H:%M NZST/NZDT")
        md_content += f"---\n*Page generated on: {timestamp}*\n"

        # Write to a .md file using only the basin_name
        filename = output_dir / f"{basin_name}.md"
        with filename.open("w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"Generated {filename}")

    # Print the number of files generated (number of unique basin names)
    print(f"Generated {len(basin_versions)} .md files in {output_dir}")


if __name__ == "__main__":
    app()
