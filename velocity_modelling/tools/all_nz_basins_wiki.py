"""
Generates a Markdown file summarizing basins defined in a YAML model version.
It requires a nationwide basin map and a CSV mapping of basins to regions processed by "map_all_basins.py"

Usage:
    python all_nz_basins_wiki.py <model_version_file> <basin_map_file> <basin_region_csv>

"""

import re
from pathlib import Path
from typing import Annotated

import typer
import yaml

app = typer.Typer(pretty_exceptions_enable=False)


def parse_basin_entry(basin_entry: str) -> tuple[str, str]:
    """
    Splits basin name and version.
    Handles entries like "Wellington_v21p8" to return ("Wellington", "v21p8").
    If no version is found, returns the entry as is with an empty version.

    Parameters
    ----------
    basin_entry : str
        The basin entry string, e.g., "Wellington_v21p8".

    Returns
    -------
    tuple[str, str]
        A tuple containing the base name and version.
        If no version is found, returns the entry as the base name and an empty string for version.
    """

    match = re.match(r"(.+)_v\d+p\d+", basin_entry)
    if match:
        base_name = match.group(1)
        version = basin_entry.split("_")[-1]
        return base_name, version
    else:
        return basin_entry, ""


def load_basin_region_mapping(mapping_file: Path) -> dict[str, str]:
    """
    Loads a CSV mapping of basin names to regions.

    Each line should be formatted as "name,region".
    Lines that are empty or start with '#' are ignored.
    If a basin name is not found in the mapping, it will be categorized as "Uncategorized" by the calling function.

    Parameters
    ----------
    mapping_file : Path
        Path to the CSV file containing basin to region mappings.

    Raises
    ------
    FileNotFoundError
        If the mapping file does not exist.
    ValueError
        If a line in the mapping file is malformed (e.g., does not contain a comma,
        or has an empty name or region).

    Returns
    -------
    dict[str, str]
        A dictionary mapping basin names to regions.
    """

    region_map = {}
    try:
        with open(mapping_file, "r") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    name, region = line.split(",", 1)
                except ValueError:
                    raise ValueError(
                        f"Malformed line #{i} in mapping file: {line}\n"
                        "Each data line must contain a comma to separate the basin name and region."
                    )

                name = name.strip()
                region = region.strip()

                if not name or not region:
                    raise ValueError(
                        f"Malformed line #{i} in mapping file: {line}\n"
                        "Basin name and region cannot be empty."
                    )

                region_map[name] = region

    except FileNotFoundError:
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    return region_map


@app.command()
def generate_basin_markdown(
    model_version_file: Annotated[
        Path, typer.Argument(help="Path to the model version YAML file", exists=True)
    ],
    basin_map_file: Annotated[
        Path,
        typer.Argument(
            help="Relative path to the PNG map of basins from the wiki to be generated",
            exists=True,
        ),
    ],
    basin_region_csv: Annotated[
        Path,
        typer.Argument(
            help="CSV mapping of basin to region (name,region)", exists=True
        ),
    ],
) -> None:
    """
    Generates a Markdown file summarizing basins by geographic region.
    The model version file should be a YAML file containing basin definitions.
    The basin map file should be a PNG image showing the basins.
    The basin region CSV should map basin names to regions in the format "name,region".

    Parameters
    ----------
    model_version_file : Path
        Path to the YAML file containing model version data with basin definitions.
    basin_map_file : Path
        Path to the PNG image file showing the basin map.
    basin_region_csv : Path
        Path to the CSV file mapping basin names to regions in the format "name,region".

    """
    with open(model_version_file, "r") as f:
        model_data = yaml.safe_load(f)

    model_name = model_version_file.stem
    model_version = (
        model_name.replace("p", ".")
        if "p" in model_name and model_name.replace("p", "").isdigit()
        else model_name
    )

    basin_entries = model_data.get("basins", [])
    if not basin_entries:
        typer.echo("❌ No basins found in the YAML file.", err=True)
        raise typer.Exit(1)

    region_lookup = load_basin_region_mapping(basin_region_csv)
    region_basin_map: dict[str, list[tuple[str, str]]] = {}

    for entry in basin_entries:
        base, version = parse_basin_entry(entry)
        region = region_lookup.get(base, "Uncategorized")
        region_basin_map.setdefault(region, []).append((base, version))

    output_file = Path("Basins.md")
    with open(output_file, "w") as out:
        out.write(
            f"# Basins in the New Zealand Velocity Model (version {model_version})\n\n"
        )
        out.write(
            "This page provides an overview of sedimentary basin models integrated into the New Zealand Velocity Model.\n\n"
        )
        out.write("<!-- Referenced map image -->\n")
        out.write(
            f'<img src="{basin_map_file}" width="50%" alt="New Zealand basins overview">\n\n'
        )

        # Write grouped content by region only
        for subregion in sorted(region_basin_map):
            out.write(f"## {subregion}\n")
            for base, version in sorted(region_basin_map[subregion]):
                out.write(f"- [{base} {version}](basins/{base}.md)\n")
            out.write("\n")

        out.write("## Version Information\n\n")
        out.write(
            "The version number in each basin name (e.g., v19p1) follows the format:\n"
        )
        out.write("- Year (e.g., 19 = 2019)\n")
        out.write("- Point release (e.g., p1 = January)\n\n")
        out.write(
            "Newer versions typically include refinements to basin geometry or velocity structure.\n"
        )

    typer.echo(f"✅ Markdown file written: {output_file}")


if __name__ == "__main__":
    app()
