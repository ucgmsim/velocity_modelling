"""
generate_thresholds.py

Command-line interface for computing threshold velocity/depth values (Vs30, Vs500, Z1.0, Z2.5)
for station locations using the NZCVM Python implementation.

This script provides a CLI wrapper around the core compute_station_thresholds() function
from threshold.py. For programmatic access, import and use compute_station_thresholds() directly.

Usage
-----
Command line:
    python generate_thresholds.py stations.ll --model-version 2.07 --topo-type SQUASHED_TAPERED

Python code:
    from velocity_modelling.threshold import compute_station_thresholds, VSType
    results_df = compute_station_thresholds(stations_df, vs_types=[VSType.VS30])

Notes
-----
- Output CSV includes Station_Name as index and computed threshold columns (and sigma if applicable).
  Longitude and latitude columns are not included in the output.
- Default behaviour computes Z1.0 and Z2.5 with SQUASHED topography.
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
from velocity_modelling.threshold import VSType, compute_station_thresholds

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nzcvm")

app = typer.Typer(pretty_exceptions_enable=False)


def read_station_file(station_file: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Read a station file and return station coordinates.

    Parameters
    ----------
    station_file : Path
        Path to the station file (format: lon lat station_name).
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
        If the station file does not exist.
    ValueError
        If the station file cannot be parsed.
    """
    if not station_file.exists():
        raise FileNotFoundError(f"Station file not found: {station_file}")

    try:
        df = pd.read_csv(
            station_file, header=None, sep=r"\s+", names=["lon", "lat", "Station_Name"]
        ).set_index("Station_Name")
        return df
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise ValueError(f"Failed to read station file {station_file}: {str(e)}")


@cli.from_docstring(app)
def generate_thresholds(
    station_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
        ),
    ],
    model_version: str = "2.07",
    vs_type: list[VSType] | None = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
        ),
    ] = None,
    topo_type: str = TopoTypes.SQUASHED.name,
    no_header: bool = False,
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

    Parameters
    ----------
    station_file : Path
        Station file path (format: lon lat station_name).
    model_version : str
        NZCVM model version to use (default: "2.07").
    vs_type : list[VSType] | None
        Threshold types to compute. If None, computes [Z1.0, Z2.5].
    out_dir : Path | None
        Output directory (default: current working directory).
    topo_type : str
        Topography type (default: "SQUASHED"). Options: TRUE, BULLDOZED, SQUASHED, SQUASHED_TAPERED.
    no_header : bool
        If True, write CSV without header row.
    nzcvm_registry : Path | None
        Optional path to nzcvm_registry.yaml; defaults to registry under data root.
    nzcvm_data_root : Path | None
        Override for data root directory; if None, use configured default.
    log_level : str
        Logging level, e.g., "INFO", "DEBUG".

    Returns
    -------
    None
        Writes results to CSV named {station_file}.csv in the output directory.

    Raises
    ------
    FileNotFoundError
        If registry or station file cannot be found.
    ValueError
        For invalid inputs or configuration issues.
    RuntimeError
        If model data loading or per-station processing fails.
    OSError
        If output directory cannot be created or output file cannot be written.
    """
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logger.setLevel(numeric_level)

    start_time = time.time()

    # Read station file (propagate FileNotFoundError/ValueError from read_station_file)
    stations_df = read_station_file(station_file, logger)

    logger.log(logging.INFO, f"Read {len(stations_df)} stations from {station_file}")

    # Set output directory and file
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir = out_dir.resolve()

    output_file = out_dir / station_file.with_suffix(".csv").name
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
            vs_types=vs_type,
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
            header=not no_header,
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
