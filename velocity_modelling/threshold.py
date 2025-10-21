"""
threshold.py

Shared utility functions for threshold velocity calculations (VS30, VS500, Z1.0, Z2.5).
Used by generate_thresholds.py (CLI script) and can be imported directly for programmatic use.

This module provides:
- Depth parameter lookup for different threshold type values
- Time-averaged velocity computation (VS30/VS500)
- Depth-to-threshold computation (Z1.0/Z2.5)
- Core computation function for batch threshold calculation
- File writing utilities

Usage Examples
--------------

Example 1: Basic usage with default parameters (Z1.0 and Z2.5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import pandas as pd
>>> from velocity_modelling.threshold import compute_station_thresholds
>>>
>>> # Create station DataFrame
>>> stations = pd.DataFrame({
...     'lon': [174.7762, 174.8851, 174.7725],
...     'lat': [-41.2865, -41.2148, -41.3010]
... }, index=['WGTN', 'WELC', 'SUNS'])
>>>
>>> # Compute thresholds (defaults to Z1.0 and Z2.5)
>>> results = compute_station_thresholds(stations)
>>> print(results)
          Z1.0(km)  Z2.5(km)  sigma
WGTN          0.152      0.845    0.3
WELC          0.089      0.512    0.3
SUNS          0.201      1.102    0.3


Example 2: Compute VS30 and VS500
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import pandas as pd
>>> from velocity_modelling.threshold import compute_station_thresholds, ThresholdTypes
>>>
>>> stations = pd.DataFrame({
...     'lon': [174.7762, 174.8851],
...     'lat': [-41.2865, -41.2148]
... }, index=['WGTN', 'WELC'])
>>>
>>> # Compute VS30 and VS500 only
>>> results = compute_station_thresholds(
...     stations,
...     threshold_types=[ThresholdTypes.VS30, ThresholdTypes.VS500],
...     include_sigma=False
... )
>>> print(results)
          Vs30(m/s)  Vs500(m/s)
WGTN           425.3        512.8
WELC           380.1        478.9


Example 3: Compute all threshold types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import pandas as pd
>>> from velocity_modelling.threshold import compute_station_thresholds, ThresholdTypes
>>>
>>> stations = pd.DataFrame({
...     'lon': [174.7762],
...     'lat': [-41.2865]
... }, index=['WGTN'])
>>>
>>> # Compute all threshold types
>>> results = compute_station_thresholds(
...     stations,
...     threshold_types=[ThresholdTypes.VS30, ThresholdTypes.VS500, ThresholdTypes.Z1_0, ThresholdTypes.Z2_5]
... )
>>> print(results)
      Vs30(m/s)  Vs500(m/s)  Z1.0(km)  Z2.5(km)  sigma
WGTN       425.3        512.8      0.152      0.845    0.3


Example 4: Read from station file and compute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import pandas as pd
>>> from pathlib import Path
>>> from velocity_modelling.threshold import compute_station_thresholds, ThresholdTypes
>>> from qcore.shared import get_stations
>>>
>>> # Read station file (format: lon lat station_name)
>>> station_file = Path("stations.ll")
>>> stations, lats, lons = get_stations(str(station_file), locations=True)
>>>
>>> # Create DataFrame
>>> stations_df = pd.DataFrame(
...     {"lon": lons, "lat": lats},
...     index=stations
... )
>>>
>>> # Compute thresholds
>>> results = compute_station_thresholds(
...     stations_df,
...     threshold_types=[ThresholdTypes.Z1_0, ThresholdTypes.Z2_5],
...     model_version="2.07"
... )
>>>
>>> # Save to CSV
>>> results.to_csv("threshold_results.csv")


Example 5: Custom configuration with topography and logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import pandas as pd
>>> import logging
>>> from pathlib import Path
>>> from velocity_modelling.threshold import compute_station_thresholds, ThresholdTypes
>>> from velocity_modelling.constants import TopoTypes
>>>
>>> # Create custom logger
>>> logger = logging.getLogger("my_analysis")
>>> logger.setLevel(logging.DEBUG)
>>> handler = logging.StreamHandler()
>>> logger.addHandler(handler)
>>>
>>> stations = pd.DataFrame({
...     'lon': [174.7762, 174.8851],
...     'lat': [-41.2865, -41.2148]
... }, index=['WGTN', 'WELC'])
>>>
>>> # Compute with custom settings including topography
>>> results = compute_station_thresholds(
...     stations_df=stations,
...     threshold_types=[ThresholdTypes.Z1_0, ThresholdTypes.Z2_5],
...     model_version="2.07",
...     topo_type=TopoTypes.SQUASHED_TAPERED,
...     data_root=Path("/custom/path/to/nzcvm/data"),
...     nzcvm_registry=Path("/custom/path/to/registry.yaml"),
...     logger=logger,
...     include_sigma=True,
...     show_progress=True
... )


Example 6: Integration with existing workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> import pandas as pd
>>> import numpy as np
>>> from velocity_modelling.threshold import compute_station_thresholds, ThresholdTypes
>>>
>>> # Load stations from your existing data source
>>> stations = pd.DataFrame({
...     'lon': np.array([174.7762, 174.8851, 174.7725]),
...     'lat': np.array([-41.2865, -41.2148, -41.3010]),
...     'elevation': [10, 5, 15],  # Extra columns are OK
...     'network': ['NZ', 'NZ', 'NZ']
... }, index=['WGTN', 'WELC', 'SUNS'])
>>>
>>> # Compute thresholds (only needs lon/lat columns)
>>> thresholds = compute_station_thresholds(
...     stations[['lon', 'lat']],
...     threshold_types=[ThresholdTypes.VS30, ThresholdTypes.Z1_0]
... )
>>>
>>> # Merge with original station data
>>> combined = stations.join(thresholds)
>>> print(combined)
>>>
>>> # Further analysis...
>>> vs30_mean = combined['Vs30(m/s)'].mean()
>>> print(f"Mean VS30: {vs30_mean:.1f} m/s")
"""

import logging
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from velocity_modelling.basin_model import (
    BasinMembership,
    InBasin,
    PartialBasinSurfaceDepths,
    compute_sigma_for_stations,
)
from velocity_modelling.constants import TopoTypes, get_data_root
from velocity_modelling.geometry import (
    MeshVector,
    PartialGlobalMesh,
    gen_full_model_grid_great_circle,
)
from velocity_modelling.global_model import PartialGlobalSurfaceDepths
from velocity_modelling.registry import CVMRegistry
from velocity_modelling.velocity3d import QualitiesVector

# Constants for sigma calculation
IN_BASIN_SIGMA = 0.3
OUT_BASIN_SIGMA = 0.5


class ThresholdTypes(str, Enum):
    """Enumeration of supported threshold velocity types."""

    VS30 = "30"
    VS500 = "500"
    Z1_0 = "1.0"
    Z2_5 = "2.5"


def get_depth_parameters(threshold_type: ThresholdTypes) -> tuple[float, float, float]:
    """
    Get depth parameters (zmax, zmin, h_depth) for a given ThresholdType.

    These parameters define the depth range and spacing for computing the
    velocity profile needed to calculate the threshold metric.

    Parameters
    ----------
    threshold_type : ThresholdTypes
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
        If threshold_type is not recognized.

    Examples
    --------
    >>> zmax, zmin, h_depth = get_depth_parameters(ThresholdTypes.VS30)
    >>> print(f"VS30: {zmin} to {zmax} km, spacing {h_depth} km")
    VS30: -0.0005 to 0.0305 km, spacing 0.001 km
    """
    depth_params = {
        ThresholdTypes.VS500: (0.505, -0.005, 0.01),
        ThresholdTypes.VS30: (0.0305, -0.0005, 0.001),
        ThresholdTypes.Z1_0: (2.0, 0.0, 0.01),
        ThresholdTypes.Z2_5: (20.0, 0.0, 0.05),
    }

    if threshold_type not in depth_params:
        raise ValueError(f"Threshold type '{threshold_type}' not recognized")

    return depth_params[threshold_type]


def get_output_column_name(threshold_type: ThresholdTypes) -> str:
    """
    Map a threshold type to its output column name.

    Parameters
    ----------
    threshold_type : ThresholdTypes
        The threshold type to compute.

    Returns
    -------
    str
        Output column header corresponding to the threshold type.

    Examples
    --------
    >>> get_output_column_name(ThresholdTypes.VS30)
    'Vs30(m/s)'
    >>> get_output_column_name(ThresholdTypes.Z1_0)
    'Z1.0(km)'
    """
    if threshold_type in [ThresholdTypes.VS30, ThresholdTypes.VS500]:
        return f"Vs{threshold_type.value}(m/s)"
    else:
        return f"Z{threshold_type.value}(km)"


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

    # Vs_avg = total_depth / sum(dz/Vs), converted to m/s
    vs_avg = total_depth / vs_sum * 1000.0

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
            f"Basin may be deeper than model extent ({abs(partial_global_mesh.z[-1]) / 1000:.2f} km)."
        )

    # Convert depth to kilometers (z_write is negative in meters)
    z_km = z_write / -1000.0
    return z_km


def compute_station_thresholds(
    stations_df: pd.DataFrame,
    threshold_types: list[ThresholdTypes] | None = None,
    model_version: str = "2.09",
    topo_type: TopoTypes = TopoTypes.SQUASHED,
    data_root: Path | None = None,
    nzcvm_registry: Path | None = None,
    logger: logging.Logger | None = None,
    include_sigma: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute threshold values (Vs30, Vs500, Z1.0, Z2.5) for station locations.

    This is the core computation function that can be called from scripts or
    imported for programmatic use. It computes requested thresholds for each
    station using the NZCVM velocity model.

    Parameters
    ----------
    stations_df : pd.DataFrame
        DataFrame with station coordinates. Must have:
        - Index: Station names (or any identifier)
        - Columns: 'lon' (longitude) and 'lat' (latitude)
    threshold_types : list[ThresholdTypes] | None
        List of threshold types to compute. If None, computes [Z1.0, Z2.5].
        Options: [ThresholdTypes.VS30, ThresholdTypes.VS500, ThresholdTypes.Z1_0, ThresholdTypes.Z2_5]
    model_version : str
        NZCVM model version to use (default: "2.09").
    topo_type : TopoTypes
        Topography handling method (default: TopoTypes.SQUASHED).
        Options: TRUE, BULLDOZED, SQUASHED, SQUASHED_TAPERED.
    data_root : Path | None
        Path to NZCVM data root directory. If None, uses configured default.
    nzcvm_registry : Path | None
        Path to nzcvm_registry.yaml. If None, uses {data_root}/nzcvm_registry.yaml.
    logger : logging.Logger | None
        Logger instance for progress reporting. If None, creates a basic logger.
    include_sigma : bool
        If True and computing Z-thresholds, include sigma column in output.
        Sigma represents basin membership uncertainty (0.3 in-basin, 0.5 out-basin).
    show_progress : bool
        If True, display progress bars during computation.

    Returns
    -------
    pd.DataFrame
        DataFrame with computed threshold values:
        - Index: Same as input stations_df
        - Columns: Threshold values (e.g., 'Vs30(m/s)', 'Z1.0(km)')
                  and optionally 'sigma' column

    Raises
    ------
    FileNotFoundError
        If registry file cannot be found.
    ValueError
        For invalid inputs or configuration issues.
    RuntimeError
        If model data loading or per-station processing fails.

    Notes
    -----
    - Computation time scales linearly with number of stations and threshold types
    - Each station requires loading a 1D velocity profile
    - Basin membership is pre-computed for all stations for efficiency
    - Sigma values are only meaningful for Z-type thresholds
    - Topography type affects how surface elevations are handled in the velocity model
    """
    # Set up logger if not provided
    if logger is None:
        logger = logging.getLogger("nzcvm.threshold")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    # Validate input DataFrame
    if not isinstance(stations_df, pd.DataFrame):
        raise ValueError("stations_df must be a pandas DataFrame")
    if "lon" not in stations_df.columns or "lat" not in stations_df.columns:
        raise ValueError("stations_df must have 'lon' and 'lat' columns")
    if len(stations_df) == 0:
        raise ValueError("stations_df is empty")

    # Default to Z1.0 and Z2.5 if no types specified
    if threshold_types is None or len(threshold_types) == 0:
        threshold_types = [ThresholdTypes.Z1_0, ThresholdTypes.Z2_5]
        logger.log(
            logging.INFO, "No threshold_types specified, defaulting to Z1.0 and Z2.5"
        )
    else:
        threshold_types = list(threshold_types)

    logger.log(
        logging.INFO,
        f"Starting threshold calculation for: {', '.join([t.value for t in threshold_types])}",
    )
    logger.log(logging.INFO, f"Model version: {model_version}")
    logger.log(logging.INFO, f"Number of stations: {len(stations_df)}")

    # Get data root
    data_root = get_data_root(data_root)
    logger.log(logging.INFO, f"Using data root: {data_root}")

    # Validate registry
    registry_path = (
        nzcvm_registry if nzcvm_registry else data_root / "nzcvm_registry.yaml"
    )
    if not registry_path.exists():
        error_msg = f"Registry file not found: {registry_path}"
        logger.log(logging.ERROR, error_msg)
        raise FileNotFoundError(error_msg)
    logger.log(logging.INFO, f"Using registry: {registry_path}")

    # Initialize registry
    logger.log(logging.INFO, "Initializing velocity model registry")
    try:
        cvm_registry = CVMRegistry(model_version, data_root, nzcvm_registry, logger)
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        logger.log(logging.ERROR, f"Failed to initialize CVM registry: {e}")
        raise RuntimeError(f"Failed to initialize CVM registry: {str(e)}") from e

    # Load all required data once
    logger.log(logging.INFO, "Loading velocity model data")
    try:
        vm1d_data, nz_tomography_data, global_surfaces, basin_data_list = (
            cvm_registry.load_all_global_data()
        )
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        logger.log(logging.ERROR, f"Failed to load model data: {e}")
        raise RuntimeError(f"Failed to load model data: {str(e)}")

    # Basin membership pre-processing
    logger.log(logging.INFO, "Pre-computing basin membership for all stations")
    basin_membership = BasinMembership(
        basin_data_list,
        smooth_boundary=nz_tomography_data.smooth_boundary,
        logger=logger,
    )

    # Pre-compute basin membership for all stations
    station_basin_membership = basin_membership.check_stations(
        stations_df["lat"].values, stations_df["lon"].values
    )

    logger.log(
        logging.INFO, f"Basin membership computed for {len(stations_df)} stations"
    )

    # Compute sigma values once (for Z-type thresholds)
    sigma_values = None
    if include_sigma and any(
        vt in [ThresholdTypes.Z1_0, ThresholdTypes.Z2_5] for vt in threshold_types
    ):
        sigma_values = compute_sigma_for_stations(
            station_basin_membership, IN_BASIN_SIGMA, OUT_BASIN_SIGMA
        )
        n_in_basin = np.sum(sigma_values == IN_BASIN_SIGMA)
        n_out_basin = np.sum(sigma_values == OUT_BASIN_SIGMA)
        logger.log(
            logging.INFO,
            f"Sigma values computed: {n_in_basin} stations in basin (σ={IN_BASIN_SIGMA}), "
            f"{n_out_basin} stations outside basin (σ={OUT_BASIN_SIGMA})",
        )

    # Initialize results DataFrame (copy to preserve input)
    results_df = pd.DataFrame(index=stations_df.index)

    # Process each threshold type
    for threshold_type_current in threshold_types:
        logger.log(logging.INFO, f"Processing {threshold_type_current.value}")

        # Get depth parameters for this threshold type
        zmax, zmin, h_depth = get_depth_parameters(threshold_type_current)

        # Set up model parameters template
        vm_params_template = {
            "model_version": model_version,
            "origin_rot": 0,
            "extent_x": 1,
            "extent_y": 1,
            "extent_zmax": zmax,
            "extent_zmin": zmin,
            "h_depth": h_depth,
            "h_lat_lon": 1,
            "topo_type": topo_type,
        }

        nx = 1
        ny = 1
        nz = int((zmax - zmin) / h_depth + 0.5)

        vm_params_template["nx"] = nx
        vm_params_template["ny"] = ny
        vm_params_template["nz"] = nz

        # Storage for threshold values for this type
        threshold_values = []

        # Set up progress bar iterator
        iterator = stations_df.iterrows()
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(stations_df),
                desc=f"Computing {threshold_type_current.value}",
            )

        # Process each station
        for idx, (station_name, row) in enumerate(iterator):
            lat = row["lat"]
            lon = row["lon"]

            try:
                # Create model parameters for this specific location
                vm_params = vm_params_template.copy()
                vm_params["origin_lat"] = lat
                vm_params["origin_lon"] = lon

                # Generate simple mesh for this single point
                global_mesh = gen_full_model_grid_great_circle(vm_params, None)

                # Use pre-computed basin membership
                basin_indices = station_basin_membership[idx]

                # Create simple InBasin objects without mesh preprocessing
                in_basin_list = [
                    InBasin(basin_data, len(global_mesh.z))
                    for basin_data in basin_data_list
                ]

                # Mark basin membership using pre-computed results
                for basin_id in basin_indices:
                    if basin_id >= 0:
                        in_basin_list[basin_id].in_basin_lat_lon = True

                # Create a minimal PartialGlobalMesh for this point
                partial_global_mesh = PartialGlobalMesh(global_mesh, 0)
                mesh_vector = MeshVector(partial_global_mesh, 0)

                # Initialize surface depths
                partial_global_surface_depths = PartialGlobalSurfaceDepths(
                    len(global_surfaces)
                )
                partial_basin_surface_depths = [
                    PartialBasinSurfaceDepths(basin_data)
                    for basin_data in basin_data_list
                ]

                # Initialize qualities vector
                qualities_vector = QualitiesVector(partial_global_mesh.nz)

                # Assign velocity/density properties
                qualities_vector.assign_qualities(
                    cvm_registry,
                    vm1d_data,
                    nz_tomography_data,
                    global_surfaces,
                    basin_data_list,
                    mesh_vector,
                    partial_global_surface_depths,
                    partial_basin_surface_depths,
                    in_basin_list,
                    basin_membership,
                    vm_params["topo_type"],
                )

                # Calculate threshold value
                if threshold_type_current in [
                    ThresholdTypes.VS30,
                    ThresholdTypes.VS500,
                ]:
                    threshold_value = compute_vs_average(
                        partial_global_mesh, qualities_vector
                    )
                else:  # Z1.0 or Z2.5
                    if threshold_type_current == ThresholdTypes.Z1_0:
                        z_thresh = 1.0
                    elif threshold_type_current == ThresholdTypes.Z2_5:
                        z_thresh = 2.5
                    else:
                        raise ValueError(
                            f"THRESHOLD_TYPE {threshold_type_current} is not a Z-type threshold"
                        )

                    threshold_value = compute_z_threshold(
                        partial_global_mesh, qualities_vector, z_thresh
                    )

                threshold_values.append(threshold_value)

            except (ValueError, KeyError, RuntimeError, OSError) as e:
                logger.log(
                    logging.ERROR,
                    f"Error processing station {station_name} at ({lat}, {lon}): {e}",
                )
                raise RuntimeError(
                    f"Failed to process station {station_name} at ({lat}, {lon}): {str(e)}"
                )

        # Add threshold values to results DataFrame
        column_name = get_output_column_name(threshold_type_current)
        results_df[column_name] = threshold_values

        logger.log(
            logging.INFO,
            f"Completed {threshold_type_current.value} calculation for {len(stations_df)} stations",
        )

    # Add sigma column if computed
    if sigma_values is not None:
        results_df["sigma"] = sigma_values
        logger.log(logging.INFO, "Added sigma values to results")

    logger.log(logging.INFO, "Threshold calculation complete")
    return results_df
