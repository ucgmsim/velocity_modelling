"""Utilities for reading and writing 1-D velocity profiles in different formats."""

from pathlib import Path

import numpy as np
import pandas as pd


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
    required_columns = {'width', 'Vp', 'Vs', 'rho', 'Qp', 'Qs'}
    missing_columns = required_columns - set(velocity_model_1d.columns.tolist())
    if missing_columns:
        raise ValueError(f'Missing required columns: {", ".join(missing_columns)}')
    
    # Check for negative values
    if np.any(velocity_model_1d[list(required_columns)] < 0):
        raise ValueError('Velocity model may not contain negative numbers.')
    
    velocity_model_1d['top_depth'] = velocity_model_1d['width'].cumsum() - velocity_model_1d['width']
    velocity_model_1d['bottom_depth'] = velocity_model_1d['width'] + velocity_model_1d['top_depth']
    return velocity_model_1d

def read_velocity_model_1d_plain_text(velocity_model_path: Path) -> pd.DataFrame:
    """Read a 1D velocity model from a plain text file and compute depth boundaries.
    
    Parameters
    ----------
    velocity_model_path : Path
        Path to the text file containing the velocity model data.
        File format must be:
        - First line: number of layers (integer)
        - Following lines: space-separated values for width, Vp, Vs, rho, Qp, Qs
    
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
        If the number of layers in the file doesn't match the header count
        If any values in the velocity model are negative
        If the header value is not a positive integer
    
    Notes
    -----
    The file format is compatible with the genslip program and related tools.
    Layer depths are computed by accumulating the width values, where each layer's
    top depth is the sum of all previous layer widths.
    """
    with open(velocity_model_path, 'r') as velocity_model:
        # Validate header
        try:
            num_layers = int(next(velocity_model))
            if num_layers <= 0:
                raise ValueError('Number of layers must be positive.')
        except (ValueError, StopIteration) as e:
            raise ValueError('Invalid or missing layer count in header.') from e
            
        # Read data
        try:
            velocity_model_df = pd.read_csv(
                velocity_model, 
                header=None, 
                delimiter=r'\s+',
                names=['width', 'Vp', 'Vs', 'rho', 'Qp', 'Qs']
            )
        except pd.errors.ParserError:
            raise ValueError('Invalid data format. Expected 6 space-separated numeric values per line.')

    if len(velocity_model_df) != num_layers:
        raise ValueError('Number of velocity model layers does not match the header.')

    if np.any(velocity_model_df < 0):
        raise ValueError('Velocity model may not contain negative numbers.')
    
    velocity_model_df['top_depth'] = velocity_model_df['width'].cumsum() - velocity_model_df['width']
    velocity_model_df['bottom_depth'] = velocity_model_df['width'] + velocity_model_df['top_depth']
    return velocity_model_df

def write_velocity_model_1d_plain_text(velocity_model: pd.DataFrame, output_path: Path) -> None:
    """Write a 1D velocity model to a plain text file in a specific format.
    
    Parameters
    ----------
    velocity_model : pd.DataFrame
        DataFrame containing the velocity model data.
        Must contain the following columns:
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
    
    output_path : Path
        Path where the output file will be written.
        
    Raises
    ------
    KeyError
        If any required columns are missing from the input DataFrame
    ValueError
        If any values in the velocity model are negative
    """
    required_columns = ['width', 'Vp', 'Vs', 'rho', 'Qp', 'Qs']
    
    with open(output_path, 'w') as output_file:
        output_file.write(f'{len(velocity_model)}\n')
        velocity_model[required_columns].to_csv(output_file, sep=' ', header=False, index=False)
