from typing import Annotated
from pathlib import Path
import typer
import yaml
import re

app = typer.Typer(pretty_exceptions_enable=False)

def parse_basin_entry(basin_entry: str) -> tuple[str, str]:
    """
    Splits basin name and version.

    Returns:
        (base_name, version) such as ("Wellington", "v21p8")
    """
    match = re.match(r"(.+)_v\d+p\d+", basin_entry)
    if match:
        base_name = match.group(1)
        version = basin_entry.split("_")[-1]
        return base_name, version
    else:
        return basin_entry, ""


def load_basin_region_mapping(mapping_file: Path) -> dict[str, str]:
    region_map = {}
    with open(mapping_file, "r") as f:
        for line in f:
            if "," in line:
                name, region = line.strip().split(",", 1)
                region_map[name] = region
    return region_map


@app.command()
def generate_basin_markdown(
    model_version_file: Annotated[
        Path,
        typer.Argument(help="Path to the model version YAML file", exists=True)
    ],
    basin_map_file: Annotated[
        Path,
        typer.Argument(help="Relative path to the PNG map of basins from the wiki to be generated", exists=True)
    ],
    basin_region_csv: Annotated[
        Path,
        typer.Argument(help="CSV mapping of basin to region (name,region)", exists=True)
    ]
) -> None:
    """
    Generates a Markdown file summarizing basins by geographic region.
    """
    with open(model_version_file, "r") as f:
        model_data = yaml.safe_load(f)

    model_name = model_version_file.stem
    model_version = model_name.replace("p", ".") if "p" in model_name and model_name.replace("p","").isdigit() else model_name

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

    output_file = Path(f"Basins.md")
    with open(output_file, "w") as out:
        out.write(f"# Basins in the New Zealand Velocity Model (version {model_version})\n\n")
        out.write("This page provides an overview of sedimentary basin models integrated into the New Zealand Velocity Model.\n\n")
        out.write(f"<!-- Referenced map image -->\n")
        out.write(f'<img src="{basin_map_file}" width="50%" alt="New Zealand basins overview">\n\n')

        # Write grouped content by region only
        for subregion in sorted(region_basin_map):
            out.write(f"## {subregion}\n")
            for base, version in sorted(region_basin_map[subregion]):
                out.write(f"- [{base} {version}](basins/{base}.md)\n")
            out.write("\n")

        out.write("## Version Information\n\n")
        out.write("The version number in each basin name (e.g., v19p1) follows the format:\n")
        out.write("- Year (e.g., 19 = 2019)\n")
        out.write("- Point release (e.g., p1 = January)\n\n")
        out.write("Newer versions typically include refinements to basin geometry or velocity structure.\n")

    typer.echo(f"✅ Markdown file written: {output_file}")


if __name__ == "__main__":
    app()
