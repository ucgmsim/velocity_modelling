"""
1D Velocity Model Module.

This module provides the VelocityModel1D class for representing one-dimensional
velocity-depth profiles. These profiles define seismic properties (P-wave velocity,
S-wave velocity, density and quality factors (Qp, Qs)) as a function of depth.

The 1D velocity models serve as reference profiles for 3D velocity modeling and
are used as baseline models in regions where more detailed information is unavailable.

Also included are utilities for reading and writing 1D velocity profiles in different formats.

There are three known formats for 1D velocity models:
1. Parquet files containing columns for 'width' and Vp, Vs, rho, Qp, Qs
2. Plain text files with the extension ".1d", containing space-separated values for width, Vp, Vs, rho, Qp, Qs
3. Plain text files with the extension ".fd_modfile", containing space-separated values for Vp, Vs, rho, Qp, Qs, bottom_depth

The read_velocity_model_1d function reads a 1D velocity model from a Parquet file and computes depth boundaries.
The read_velocity_model_1d_plain_text function reads a 1D velocity model from a plain text file and computes depth boundaries.
The write_velocity_model_1d_plain_text function writes a 1D velocity model to a plain text file in a specific format.

"""

from pathlib import Path

import numpy as np
import pandas as pd


class VelocityModel1D:
    """
    Class representing a one-dimensional velocity-depth profile loaded from a file.

    This class stores arrays of P-wave velocity, S-wave velocity, density, quality factors (Qp, Qs) and
    corresponding depths, allowing for the representation of layered earth models.

    Parameters
    ----------
    file_path : str or Path
        Path to the velocity model file. Supports .parquet, .1d, or .fd_modfile formats.

    Attributes
    ----------
    vp : np.ndarray
        P-wave velocities (km/s).
    vs : np.ndarray
        S-wave velocities (km/s).
    rho : np.ndarray
        Densities (g/cm^3).
    qp : np.ndarray
        P-wave quality factors.
    qs : np.ndarray
        S-wave quality factors.
    depth : np.ndarray
        Depths (m) of the bottom of each layer.
    n_depth : int
        Number of depth points.
    """

    def __init__(self, file_path: str | Path):
        """
        Initialize the VelocityModel1D from a file path.

        Parameters
        ----------
        file_path : str or Path
            Path to the velocity model file. Extension determines the format.
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        # Load data based on file extension
        if file_ext == ".parquet":
            df = read_velocity_model_1d(file_path)
        elif file_ext in [".1d", ".fd_modfile"]:
            df = read_velocity_model_1d_plain_text(file_path)
        else:
            raise ValueError(
                f"Unsupported file extension: {file_ext}. "
                "Supported formats: .parquet, .1d, .fd_modfile"
            )

        # Assign dataframe columns to class attributes
        self.vp = df["Vp"].values
        self.vs = df["Vs"].values
        self.rho = df["rho"].values
        self.qp = df["Qp"].values
        self.qs = df["Qs"].values
        self.bottom_depth = df["bottom_depth"].values
        self.top_depth = df["top_depth"].values
        self.width = df["width"].values
        self.n_depth = len(self.bottom_depth)


def read_velocity_model_1d(velocity_model_path: Path) -> pd.DataFrame:
    """Read a 1D velocity model from a Parquet file and compute depth boundaries.

    Parameters
    ----------
    velocity_model_path : Path
        Path to the Parquet file containing the velocity model data.
        Expected to contain columns for 'width' and velocity parameters.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the velocity model with additional columns:
        - top_depth : float
            Depth to the top of each layer, computed from cumulative widths
        - bottom_depth : float
            Depth to the bottom of each layer
        - width : float
            Layer width/thickness
        - Vp : float
            P-wave velocity
        - Vs : float
            S-wave velocity
        - rho : float
            Density
        - Qp : float
            P-wave quality factor
        - Qs : float
            S-wave quality factor

    Raises
    ------
    ValueError
        If any values in the velocity model are negative
        If required columns are missing
    """
    velocity_model_1d = pd.read_parquet(velocity_model_path)

    # Check for required columns
    required_columns = {"width", "Vp", "Vs", "rho", "Qp", "Qs"}
    missing_columns = required_columns - set(velocity_model_1d.columns.tolist())
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    # Check for negative values
    if np.any(velocity_model_1d[list(required_columns)] < 0):
        raise ValueError("Velocity model may not contain negative numbers.")

    velocity_model_1d["top_depth"] = (
        velocity_model_1d["width"].cumsum() - velocity_model_1d["width"]
    )
    velocity_model_1d["bottom_depth"] = (
        velocity_model_1d["width"] + velocity_model_1d["top_depth"]
    )
    return velocity_model_1d


def read_velocity_model_1d_plain_text(velocity_model_path: Path) -> pd.DataFrame:
    """Read a 1D velocity model from a plain text file and compute depth boundaries.

    Supports two file formats:
    1. Files ending with "1d":
       - First line: number of layers (integer)
       - Following lines: space-separated values for width, Vp, Vs, rho, Qp, Qs

    2. Files ending with "fd_modfile":
       - First line: header (skipped)
       - Following lines: space-separated values for Vp, Vs, rho, Qp, Qs, bottom_depth

    Parameters
    ----------
    velocity_model_path : Path
        Path to the text file containing the velocity model data.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the velocity model with columns:
        - top_depth : float
            Depth to the top of each layer, computed from cumulative widths
        - bottom_depth : float
            Depth to the bottom of each layer
        - width : float
            Layer width/thickness
        - Vp : float
            P-wave velocity
        - Vs : float
            S-wave velocity
        - rho : float
            Density
        - Qp : float
            P-wave quality factor
        - Qs : float
            S-wave quality factor

    Raises
    ------
    ValueError
        If the number of layers in a .1d file doesn't match the header count
        If any values in the velocity model are negative
        If the header value is not a positive integer
        If the file format is not supported

    Notes
    -----
    The .1d file format is compatible with the genslip program and related tools.
    The .fd_modfile format specifies bottom depths directly rather than widths.
    """
    # Check filename to determine format
    file_ext = velocity_model_path.suffix.lower()

    # 1d file format
    if file_ext == ".1d":
        with open(velocity_model_path, "r") as velocity_model:
            # Validate header
            try:
                num_layers = int(next(velocity_model))
                if num_layers <= 0:
                    raise ValueError("Number of layers must be positive.")
            except (ValueError, StopIteration) as e:
                raise ValueError("Invalid or missing layer count in header.") from e

            # Read data
            try:
                velocity_model_df = pd.read_csv(
                    velocity_model,
                    header=None,
                    delimiter=r"\s+",
                    names=["width", "Vp", "Vs", "rho", "Qp", "Qs"],
                )
            except pd.errors.ParserError:
                raise ValueError(
                    "Invalid data format. Expected 6 space-separated numeric values per line."
                )

        if len(velocity_model_df) != num_layers:
            raise ValueError(
                "Number of velocity model layers does not match the header."
            )

        if np.any(velocity_model_df < 0):
            raise ValueError("Velocity model may not contain negative numbers.")

        velocity_model_df["top_depth"] = (
            velocity_model_df["width"].cumsum() - velocity_model_df["width"]
        )
        velocity_model_df["bottom_depth"] = (
            velocity_model_df["width"] + velocity_model_df["top_depth"]
        )

    # fd_modfile format
    elif file_ext == ".fd_modfile":
        try:
            # Skip the first line and read the rest with the correct column order
            velocity_model_df = pd.read_csv(
                velocity_model_path,
                header=None,
                skiprows=1,  # Skip the first line "DEF HST"
                delimiter=r"\s+",
                names=["Vp", "Vs", "rho", "Qp", "Qs", "bottom_depth"],
            )
        except pd.errors.ParserError:
            raise ValueError("Invalid data format for fd_modfile.")

        if np.any(velocity_model_df[["Vp", "Vs", "rho", "Qp", "Qs"]] < 0):
            raise ValueError("Velocity model may not contain negative numbers.")

        # Calculate top_depth and width from bottom_depth
        velocity_model_df["top_depth"] = np.zeros(len(velocity_model_df))
        if len(velocity_model_df) > 1:
            velocity_model_df.loc[1:, "top_depth"] = (
                velocity_model_df["bottom_depth"].iloc[:-1].values
            )

        # Calculate width as the difference between bottom and top depths
        velocity_model_df["width"] = (
            velocity_model_df["bottom_depth"] - velocity_model_df["top_depth"]
        )

    else:
        raise ValueError(
            f"Unsupported file format: {file_ext}. Filename must end with '1d' or 'fd_modfile'"
        )

    return velocity_model_df


def write_velocity_model_1d_plain_text(
    velocity_model: pd.DataFrame, output_path: Path
) -> None:
    """Write a 1D velocity model to a plain text file.

    Supports two output formats based on file extension:
    1. Files ending with ".1d":
       - First line: number of layers (integer)
       - Following lines: space-separated values for width, Vp, Vs, rho, Qp, Qs

    2. Files ending with ".fd_modfile":
       - First line: header "DEF HST"
       - Following lines: space-separated values for Vp, Vs, rho, Qp, Qs, bottom_depth

    Parameters
    ----------
    velocity_model : pd.DataFrame
        DataFrame containing the velocity model data.
        Must contain the following columns:
        - Vp : float
            P-wave velocity
        - Vs : float
            S-wave velocity
        - rho : float
            Density
        - Qp : float
            P-wave quality factor
        - Qs : float
            S-wave quality factor
        - bottom_depth : float
            Depth to the bottom of each layer
        - width : float (for .1d format)
            Layer width/thickness

    output_path : Path
        Path where the output file will be written.

    Raises
    ------
    KeyError
        If any required columns are missing for the specified format
    ValueError
        If any values in the velocity model are negative or if the file format is unsupported
    """
    file_ext = output_path.suffix.lower()

    # Check for negative values in the core properties
    core_columns = ["Vp", "Vs", "rho", "Qp", "Qs"]
    if np.any(velocity_model[core_columns] < 0):
        raise ValueError("Velocity model may not contain negative numbers.")

    # .1d format
    if file_ext == ".1d":
        if "width" not in velocity_model.columns:
            raise KeyError("Column 'width' is required for .1d format")

        required_columns = ["width", "Vp", "Vs", "rho", "Qp", "Qs"]

        with open(output_path, "w") as output_file:
            output_file.write(f"{len(velocity_model)}\n")
            velocity_model[required_columns].to_csv(
                output_file, sep=" ", header=False, index=False
            )

    # fd_modfile format
    elif file_ext == ".fd_modfile":
        if "bottom_depth" not in velocity_model.columns:
            raise KeyError("Column 'bottom_depth' is required for .fd_modfile format")

        fd_columns = ["Vp", "Vs", "rho", "Qp", "Qs", "bottom_depth"]

        with open(output_path, "w") as output_file:
            output_file.write("DEF HST\n")
            velocity_model[fd_columns].to_csv(
                output_file, sep=" ", header=False, index=False
            )

    else:
        raise ValueError(
            f"Unsupported file extension: {file_ext}. Output filename must end with '.1d' or '.fd_modfile'"
        )
