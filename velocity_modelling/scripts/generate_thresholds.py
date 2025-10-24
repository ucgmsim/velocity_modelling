"""
generate_thresholds.py

Command-line interface for computing threshold velocity/depth values (Vs30, Vs500, Z1.0, Z2.5)
for station locations using the NZCVM Python implementation.

This script provides a CLI wrapper around the core compute_station_thresholds() function
from threshold.py. For programmatic access, import and use compute_station_thresholds() directly.

Usage
-----
Command line (default CSV format):
    python generate_thresholds.py locations.csv --model-version 2.07 --topo-type SQUASHED_TAPERED

Command line (original station file format):
    python generate_thresholds.py stations.ll --lon-index 0 --lat-index 1 --name-index 2 --sep " "

Python code:
    from velocity_modelling.threshold import compute_station_thresholds, ThresholdTypes
    results_df = compute_station_thresholds(stations_df, vs_types=[ThresholdTypes.VS30])

Notes
-----
- Output CSV includes Station_Name as index and computed threshold columns (and sigma if applicable).
  Longitude and latitude columns are not included in the output.
- Default behaviour computes Z1.0 and Z2.5 with SQUASHED topography.
- Default input format expects CSV with columns: [name, latitude, longitude]
- For legacy station files, use --lon-index, --lat-index, --name-index and --sep options
"""

import logging
import sys
import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from qcore import cli
from velocity_modelling.constants import TopoTypes
from velocity_modelling.threshold import ThresholdTypes, compute_station_thresholds

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nzcvm")

app = typer.Typer(pretty_exceptions_enable=False)


def read_locations_file(
    locations_csv: Path,
    name_index: int,
    lat_index: int,
    lon_index: int,
    sep: str,
    skip_rows: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Read a locations file and return station coordinates.

    The file can contain additional columns beyond the required name, latitude, and longitude columns.
    Only the columns specified by the index parameters will be extracted.

    Parameters
    ----------
    locations_csv : Path
        Path to the locations file.
    name_index : int
        Column index for station names/IDs (0-based).
    lat_index : int
        Column index for latitude (0-based).
    lon_index : int
        Column index for longitude (0-based).
    sep : str
        Column separator/delimiter.
    skip_rows : int
        Number of header rows to skip at the beginning of the file.
    logger : logging.Logger
        Logger instance for reporting errors.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Station_Name with columns:
        - lon: float, station longitude
        - lat: float, station latitude

    Raises
    ------
    FileNotFoundError
        If the locations file does not exist.
    ValueError
        If the locations file cannot be parsed or indices are invalid.
    """
    if not locations_csv.exists():
        raise FileNotFoundError(f"Locations file not found: {locations_csv}")

    try:
        # Read file without assuming column names, skipping header rows
        df = pd.read_csv(locations_csv, header=None, sep=sep, skiprows=skip_rows)

        # Validate column indices
        max_index = max(name_index, lat_index, lon_index)
        if max_index >= len(df.columns):
            raise ValueError(
                f"Column index {max_index} out of range. File has {len(df.columns)} columns. "
                f"Available indices: 0-{len(df.columns) - 1}. "
                f"Requested: name_index={name_index}, lat_index={lat_index}, lon_index={lon_index}"
            )

        # Extract required columns using specified indices
        station_names = df.iloc[:, name_index].astype(str)
        lats = pd.to_numeric(df.iloc[:, lat_index], errors="coerce")
        lons = pd.to_numeric(df.iloc[:, lon_index], errors="coerce")

        # Check for invalid coordinate values
        if lats.isna().any() or lons.isna().any():
            invalid_rows = df[lats.isna() | lons.isna()]
            raise ValueError(
                f"Invalid latitude/longitude values found in rows: {invalid_rows.index.tolist()}. "
                f"Check that columns at lat_index={lat_index} and lon_index={lon_index} contain numeric values."
            )

        # Create output DataFrame
        result_df = pd.DataFrame(
            {"lat": lats.values, "lon": lons.values}, index=station_names.values
        )
        result_df.index.name = "Station_Name"

        return result_df

    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise ValueError(
            f"Failed to read locations file: The data is possibly not correctly formatted. "
            f"name_index={name_index}, lat_index={lat_index}, lon_index={lon_index}, sep='{sep}', skip_rows={skip_rows}. "
        ) from e


@cli.from_docstring(app)
def generate_thresholds(
    locations_csv: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
        ),
    ],
    name_index: int = 0,
    lon_index: int = 1,
    lat_index: int = 2,
    sep: str = ",",
    skip_rows: int = 0,
    model_version: str = "2.09",
    threshold_type: list[ThresholdTypes] | None = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
        ),
    ] = None,
    write_header: bool = True,
    topo_type: str = TopoTypes.SQUASHED.name,
    nzcvm_registry: Annotated[
        Path | None,
        typer.Option(
            exists=False,
            dir_okay=False,
        ),
    ] = None,
    nzcvm_data_root: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            exists=False,
        ),
    ] = None,
    log_level: str = "INFO",
) -> None:
    """
    Generate threshold values (Vs30, Vs500, Z1.0, Z2.5) for station locations.

    This computes requested thresholds for each station, using precomputed basin
    membership for sigma assignment when Z-thresholds are requested.

    The default input format expects a CSV file with columns: [name, lon, lat].
    For legacy station files (format: lon lat name), use:
    --lon-index 0 --lat-index 1 --name-index 2 --sep " "

    Parameters
    ----------
    locations_csv : Path
        CSV file with station locations. Default format: [name, lon, lat].
    name_index : int
        Column index for station names/IDs (0-based, default: 0).
    lon_index : int
        Column index for longitude (0-based, default: 1).
    lat_index : int
        Column index for latitude (0-based, default: 2).
    sep : str
        Column separator/delimiter (default: ",").
    skip_rows : int
        Number of header rows to skip at the beginning of the locations_csv file (default: 0).
    model_version : str
        NZCVM model version to use (default: "2.09").
    threshold_type : list[ThresholdTypes] | None
        Threshold types to compute. If None, computes [Z1.0, Z2.5].
    out_dir : Path | None
        Output directory (default: current working directory).
    write_header : bool
        Write CSV output with header row (default: True). Use --no-write-header to disable.
    topo_type : str
        Topography type (default: "SQUASHED"). Options: TRUE, BULLDOZED, SQUASHED, SQUASHED_TAPERED.
    nzcvm_registry : Path | None
        Optional path to nzcvm_registry.yaml; defaults to registry under data root.
    nzcvm_data_root : Path | None
        Override for data root directory; if None, use configured default.
    log_level : str
        Logging level, e.g., "INFO", "DEBUG".

    Returns
    -------
    None
        Writes results to CSV named {locations_csv}_thresholds.csv in the output directory.

    Raises
    ------
    FileNotFoundError
        If registry or locations file cannot be found.
    ValueError
        For invalid inputs or configuration issues.
    RuntimeError
        If model data loading or per-station processing fails.
    OSError
        If output directory cannot be created or output file cannot be written.

    Examples
    --------
    Default CSV format (id, lon, lat):
        generate_thresholds locations.csv --model-version 2.09

    Legacy station file format (lon lat name):
        generate_thresholds stations.ll --lon-index 0 --lat-index 1 --name-index 2 --sep " "

    Custom format with tab separator:
        generate_thresholds data.tsv --name-index 2 --lat-index 0 --lon-index 1 --sep "\t"

    CSV with header row to skip:
        generate_thresholds locations.csv --skip-rows 1

    Write output without header row:
        generate_thresholds locations.csv --no-write-header
    """
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logger.setLevel(numeric_level)

    start_time = time.time()

    # Read locations file (propagate FileNotFoundError/ValueError from read_locations_file)
    stations_df = read_locations_file(
        locations_csv, name_index, lat_index, lon_index, sep, skip_rows, logger
    )

    logger.log(logging.INFO, f"Read {len(stations_df)} stations from {locations_csv}")

    # Set output directory and file
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir = out_dir.resolve()

    # Create output filename with "_thresholds" suffix to avoid overwriting input
    output_file = out_dir / f"{locations_csv.stem}_thresholds.csv"
    logger.log(logging.INFO, f"Output will be saved as {output_file}")

    # Create output directory
    try:
        out_dir.mkdir(exist_ok=True, parents=True)
    except OSError as e:
        logger.log(logging.ERROR, f"Failed to create output directory {out_dir}: {e}")
        raise OSError(f"Failed to create output directory {out_dir}: {str(e)}")

    # Validate and convert topo_type
    try:
        topo_type_enum = TopoTypes[topo_type.upper()]
    except KeyError:
        valid_types = [t.name for t in TopoTypes]
        logger.log(logging.ERROR, f"Invalid topo type: {topo_type}")
        raise ValueError(
            f"Invalid topo type '{topo_type}'. Valid options: {valid_types}"
        )

    # Call the core computation function
    try:
        results_df = compute_station_thresholds(
            stations_df=stations_df,
            threshold_types=threshold_type,
            model_version=model_version,
            topo_type=topo_type_enum,
            data_root=nzcvm_data_root,
            nzcvm_registry=nzcvm_registry,
            logger=logger,
            include_sigma=True,
            show_progress=True,
        )
    except Exception as e:
        logger.log(logging.ERROR, f"Threshold computation failed: {e}")
        raise

    # Write results to file
    logger.log(logging.INFO, f"Writing results to {output_file}")
    try:
        results_df.to_csv(
            output_file,
            index=True,
            index_label="Station_Name",
            header=write_header,
            float_format="%.3f",
        )
        logger.log(logging.INFO, f"Results successfully written to {output_file}")
    except (OSError, ValueError) as e:
        logger.log(logging.ERROR, f"Failed to write results: {e}")
        raise OSError(f"Failed to write results to {output_file}: {str(e)}")

    logger.log(logging.INFO, "Threshold calculation 100% complete")
    elapsed_time = time.time() - start_time
    logger.log(
        logging.INFO, f"Threshold calculation completed in {elapsed_time:.2f} seconds"
    )


if __name__ == "__main__":
    app()
