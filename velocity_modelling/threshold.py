"""
threshold.py

Shared utility functions for threshold velocity calculations (VS30, VS500, Z1.0, Z2.5).
Used by generate_threshold_grid.py and generate_threshold_points.py.

This module provides:
- Depth parameter lookup for different VS_TYPE values
- Time-averaged velocity computation (VS30/VS500)
- Depth-to-threshold computation (Z1.0/Z2.5)
- File writing utilities
"""

from enum import Enum
from pathlib import Path

import numpy as np

from velocity_modelling.geometry import PartialGlobalMesh
from velocity_modelling.velocity3d import QualitiesVector


class VSType(str, Enum):
    """Enumeration of supported threshold velocity types."""

    VS30 = "VS30"
    VS500 = "VS500"
    Z1_0 = "Z1.0"
    Z2_5 = "Z2.5"


def get_depth_parameters(vs_type: VSType) -> tuple[float, float, float]:
    """
    Get depth parameters (zmax, zmin, h_depth) for a given VS_TYPE.

    These parameters define the depth range and spacing for computing the
    velocity profile needed to calculate the threshold metric.

    Parameters
    ----------
    vs_type : VSType
        Type of velocity threshold to calculate.

    Returns
    -------
    tuple[float, float, float]
        (zmax, zmin, h_depth) in kilometers:
        - zmax: Maximum depth (positive downwards)
        - zmin: Minimum depth (negative upwards for above surface)
        - h_depth: Depth spacing between grid points

    Raises
    ------
    ValueError
        If the VS_TYPE is not recognized.

    Examples
    --------
    >>> zmax, zmin, h_depth = get_depth_parameters(VSType.VS30)
    >>> print(f"VS30: {zmin} to {zmax} km, spacing {h_depth} km")
    VS30: -0.0005 to 0.0305 km, spacing 0.001 km
    """
    depth_params = {
        VSType.VS500: (0.505, -0.005, 0.01),
        VSType.VS30: (0.0305, -0.0005, 0.001),
        VSType.Z1_0: (2.0, 0.0, 0.01),
        VSType.Z2_5: (20.0, 0.0, 0.05),
    }

    if vs_type not in depth_params:
        raise ValueError(f"VS type '{vs_type}' not recognized")

    return depth_params[vs_type]


def compute_vs_average(
    partial_global_mesh: PartialGlobalMesh,
    qualities_vector: QualitiesVector,
) -> float:
    """
    Compute time-averaged (harmonic mean) shear-wave velocity.

    The time-averaged velocity is the depth divided by the travel time
    through the layers, which gives the harmonic mean:

    Vs_avg = H / Σ(h_i / Vs_i)

    where H is total depth, h_i is layer thickness, and Vs_i is layer velocity.

    Parameters
    ----------
    partial_global_mesh : PartialGlobalMesh
        Mesh containing depth information (z values in meters, negative downwards).
    qualities_vector : QualitiesVector
        Velocity values at mesh points (Vs in km/s).

    Returns
    -------
    float
        Time-averaged Vs in km/s.

    Notes
    -----
    This calculation assumes:
    - Depths are in meters with negative values representing below surface
    - Velocities are in km/s
    - Layers have uniform spacing (dz)

    Examples
    --------
    For VS30 calculation with 30m depth and constant Vs=0.5 km/s:
    >>> vs_avg = compute_vs_average(mesh, qualities)
    >>> print(f"VS30: {vs_avg:.3f} km/s")
    """
    # Calculate dZ (spacing between depth points in meters)
    dz = partial_global_mesh.z[0] - partial_global_mesh.z[1]

    # Calculate time-averaged (harmonic mean) velocity
    # Sum of (layer_thickness / layer_velocity)
    vs_sum = 0.0
    for j in range(partial_global_mesh.nz):
        vs_sum += dz / qualities_vector.vs[j]

    # Total depth in meters (z values are negative, so we negate)
    total_depth = -partial_global_mesh.z[partial_global_mesh.nz - 1]

    # Vs_avg = total_depth / sum(dz/Vs), converted to km/s
    vs_avg = total_depth / vs_sum / 1000.0

    return vs_avg


def compute_z_threshold(
    partial_global_mesh: PartialGlobalMesh,
    qualities_vector: QualitiesVector,
    z_threshold: float,
) -> float:
    """
    Compute depth to velocity threshold.

    Finds the depth where shear-wave velocity (Vs) first exceeds the
    specified threshold value. This is commonly used for basin depth
    estimation (Z1.0, Z2.5).

    Parameters
    ----------
    partial_global_mesh : PartialGlobalMesh
        Mesh containing depth information (z values in meters, negative downwards).
    qualities_vector : QualitiesVector
        Velocity values at mesh points (Vs in km/s).
    z_threshold : float
        Threshold velocity value (km/s). Common values are 1.0 and 2.5.

    Returns
    -------
    float
        Depth to threshold in kilometers (positive downwards).

    Raises
    ------
    ValueError
        If threshold velocity is never exceeded within the depth range,
        indicating the basin is deeper than the model extent or the
        threshold is higher than the maximum velocity.

    Notes
    -----
    The function searches from the surface downward and returns the first
    depth where Vs >= threshold. If the threshold is never reached, it
    indicates the basin extends beyond the model depth or the velocity
    never reaches the threshold value.

    Examples
    --------
    >>> z_depth = compute_z_threshold(mesh, qualities, 1.0)
    >>> print(f"Depth to Vs=1.0 km/s: {z_depth:.3f} km")
    Depth to Vs=1.0 km/s: 0.152 km
    """
    z_write = 0.0

    # Search for first depth where Vs exceeds threshold
    for j in range(partial_global_mesh.nz - 1):
        if qualities_vector.vs[j] >= z_threshold:
            z_write = partial_global_mesh.z[j]
            break

    if z_write == 0:
        max_vs = max(qualities_vector.vs) if qualities_vector.vs else 0.0
        raise ValueError(
            f"Z_Threshold {z_threshold:.1f} km/s not reached within depth range. "
            f"Maximum Vs found: {max_vs:.3f} km/s. "
            f"Basin may be deeper than model extent ({abs(partial_global_mesh.z[-1])/1000:.2f} km)."
        )

    # Convert depth to kilometers (z_write is negative in meters)
    z_km = z_write / -1000.0
    return z_km


def get_output_paths(vs_type: VSType) -> tuple[str, str, str]:
    """
    Get output directory, filename, and header for a given VS_TYPE.

    Parameters
    ----------
    vs_type : VSType
        Type of velocity threshold.

    Returns
    -------
    tuple[str, str, str]
        (subdirectory, filename, header_line):
        - subdirectory: 'Vs' or 'Z'
        - filename: Name of the output file
        - header_line: Header line for the output file

    Examples
    --------
    >>> subdir, filename, header = get_output_paths(VSType.VS30)
    >>> print(subdir, filename)
    Vs Vs_30.txt
    """
    if vs_type == VSType.VS30:
        return "Vs", "Vs_30.txt", "Lon\tLat\tVs_30(km/s)\n"
    elif vs_type == VSType.VS500:
        return "Vs", "Vs_500.txt", "Lon\tLat\tVs_500(km/s)\n"
    elif vs_type == VSType.Z1_0:
        return "Z", "Z_1.0.txt", "Lon\tLat\tZ_1.0(km)\n"
    elif vs_type == VSType.Z2_5:
        return "Z", "Z_2.5.txt", "Lon\tLat\tZ_2.5(km)\n"
    else:
        raise ValueError(f"Unknown VS_TYPE: {vs_type}")


def write_threshold_value(
    output_dir: Path,
    lon: float,
    lat: float,
    threshold_value: float,
    vs_type: VSType,
    is_first: bool,
) -> None:
    """
    Write a single threshold value to the appropriate output file.

    Creates the output file on first write, appends on subsequent writes.

    Parameters
    ----------
    output_dir : Path
        Output directory.
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    threshold_value : float
        Computed threshold value.
    vs_type : VSType
        Type of threshold.
    is_first : bool
        If True, create new file with header. If False, append to existing file.

    Raises
    ------
    OSError
        If there are issues creating or writing to the output file.
    """
    subdir, filename, header = get_output_paths(vs_type)

    # Create subdirectory
    output_subdir = output_dir / subdir
    output_subdir.mkdir(exist_ok=True, parents=True)
    output_file = output_subdir / filename

    mode = "w" if is_first else "a"

    with output_file.open(mode) as f:
        if is_first:
            f.write(header)
        f.write(f"{lon:.6f}\t{lat:.6f}\t{threshold_value:.6f}\n")


def get_z_threshold_value(vs_type: VSType) -> float:
    """
    Get the velocity threshold value for Z-type calculations.

    Parameters
    ----------
    vs_type : VSType
        Type of velocity threshold.

    Returns
    -------
    float
        Threshold velocity in km/s (1.0 or 2.5).

    Raises
    ------
    ValueError
        If VS_TYPE is not a Z-type (not Z1.0 or Z2.5).

    Examples
    --------
    >>> thresh = get_z_threshold_value(VSType.Z1_0)
    >>> print(thresh)
    1.0
    """
    if vs_type == VSType.Z1_0:
        return 1.0
    elif vs_type == VSType.Z2_5:
        return 2.5
    else:
        raise ValueError(f"VS_TYPE {vs_type} is not a Z-type threshold")