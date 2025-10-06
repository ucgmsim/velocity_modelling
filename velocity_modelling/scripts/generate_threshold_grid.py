"""
generate_threshold_grid.py

This script generates threshold velocity maps (VS30, VS500, Z1.0, Z2.5) for a specified region.
It creates a 2D grid over the specified geographic area, computes velocity profiles at each grid point,
and calculates the threshold metrics (time-averaged VS or depth to velocity threshold).

This script is part of the velocity_modelling package and is designed to be run from the command line.

Usage:
    python generate_threshold_grid.py --out-dir <output_directory> --model-version <version>
                                       --origin-lat <lat> --origin-lon <lon> --origin-rot <rotation>
                                       --extent-x <x_extent> --extent-y <y_extent>
                                       --extent-latlon-spacing <spacing> --vs-type <VS30|VS500|Z1.0|Z2.5>

Example:
    python generate_threshold_grid.py --out-dir ./threshold_output --model-version 2.07
                                       --origin-lat -43.60 --origin-lon 172.30 --origin-rot -10
                                       --extent-x 140 --extent-y 120 --extent-latlon-spacing 1.0
                                       --vs-type VS30

VS_TYPE options:
    - VS30: Time-averaged shear-wave velocity in the top 30 meters
    - VS500: Time-averaged shear-wave velocity in the top 500 meters
    - Z1.0: Depth (km) to Vs = 1.0 km/s
    - Z2.5: Depth (km) to Vs = 2.5 km/s

Output:
    - Creates subdirectories 'Vs' or 'Z' in the output directory
    - Writes text file with tab-separated values: Lon, Lat, Threshold_Value
    - For VS30/VS500: Values are in km/s
    - For Z1.0/Z2.5: Values are depths in km

Example output file format (Vs/Vs_30.txt):
```
Lon	Lat	Vs_30(km/s)
172.30	-43.60	0.523
172.30	-43.61	0.547
...
```

Example output file format (Z/Z_1.0.txt):
```
Lon	Lat	Z_1.0(km)
172.30	-43.60	0.152
172.30	-43.61	0.168
...
```
"""

import logging
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from tqdm import tqdm

from qcore import cli
from velocity_modelling.basin_model import (
    InBasin,
    InBasinGlobalMesh,
    PartialBasinSurfaceDepths,
)
from velocity_modelling.constants import get_data_root
from velocity_modelling.geometry import (
    MeshVector,
    PartialGlobalMesh,
    gen_full_model_grid_great_circle,
)
from velocity_modelling.global_model import PartialGlobalSurfaceDepths
from velocity_modelling.registry import CVMRegistry
from velocity_modelling.threshold import (
    VSType,
    compute_vs_average,
    compute_z_threshold,
    get_depth_parameters,
    get_z_threshold_value,
)
from velocity_modelling.velocity3d import PartialGlobalQualities, QualitiesVector

# Configure logging at the module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("nzcvm")

app = typer.Typer(pretty_exceptions_enable=False)


def calc_and_save_vs(
    output_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities: PartialGlobalQualities,
    vs_depth: str,
    lat_ind: int,
    logger: logging.Logger,
) -> None:
    """
    Calculate time-averaged shear-wave velocity and save to file.

    Computes the harmonic mean (time-averaged) velocity over the depth range.

    Parameters
    ----------
    output_dir : Path
        Output directory where Vs subdirectory will be created.
    partial_global_mesh : PartialGlobalMesh
        Mesh containing lat, lon, and depth information.
    partial_global_qualities : PartialGlobalQualities
        Velocity and density values at mesh points.
    vs_depth : str
        Depth identifier ("30" for VS30, "500" for VS500).
    lat_ind : int
        Latitude index (0 creates new file, >0 appends).
    logger : logging.Logger
        Logger instance for logging messages.

    Raises
    ------
    OSError
        If there are issues creating or writing to the output file.
    """
    vs_dir = output_dir / "Vs"
    vs_dir.mkdir(exist_ok=True, parents=True)
    vs_file = vs_dir / f"Vs_{vs_depth}.txt"

    mode = "w" if lat_ind == 0 else "a"

    with vs_file.open(mode) as f:
        if lat_ind == 0:
            f.write(f"Lon\tLat\tVs_{vs_depth}(km/s)\n")

        for i in range(partial_global_mesh.nx):
            # Create a temporary QualitiesVector for this column
            qualities_vector = QualitiesVector(partial_global_mesh.nz)
            for j in range(partial_global_mesh.nz):
                qualities_vector.vs[j] = partial_global_qualities.vs[i][j]

            # Use shared computation function
            vs_total = compute_vs_average(partial_global_mesh, qualities_vector)

            f.write(
                f"{partial_global_mesh.lon[i]:.6f}\t"
                f"{partial_global_mesh.lat[i]:.6f}\t"
                f"{vs_total:.6f}\n"
            )

    logger.log(logging.DEBUG, f"Wrote VS{vs_depth} data to {vs_file}")


def calc_and_save_z_threshold(
    output_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities: PartialGlobalQualities,
    z_threshold: str,
    lat_ind: int,
    logger: logging.Logger,
) -> None:
    """
    Calculate depth to velocity threshold and save to file.

    Finds the depth where Vs first exceeds the threshold value.

    Parameters
    ----------
    output_dir : Path
        Output directory where Z subdirectory will be created.
    partial_global_mesh : PartialGlobalMesh
        Mesh containing lat, lon, and depth information.
    partial_global_qualities : PartialGlobalQualities
        Velocity and density values at mesh points.
    z_threshold : str
        Threshold velocity value ("1.0" or "2.5" km/s).
    lat_ind : int
        Latitude index (0 creates new file, >0 appends).
    logger : logging.Logger
        Logger instance for logging messages.

    Raises
    ------
    ValueError
        If threshold is outside depth limits.
    OSError
        If there are issues creating or writing to the output file.
    """
    z_dir = output_dir / "Z"
    z_dir.mkdir(exist_ok=True, parents=True)
    z_file = z_dir / f"Z_{z_threshold}.txt"

    z_thresh_double = float(z_threshold)

    mode = "w" if lat_ind == 0 else "a"

    with z_file.open(mode) as f:
        if lat_ind == 0:
            f.write(f"Lon\tLat\tZ_{z_threshold}(km)\n")

        for i in range(partial_global_mesh.nx):
            # Create a temporary QualitiesVector for this column
            qualities_vector = QualitiesVector(partial_global_mesh.nz)
            for j in range(partial_global_mesh.nz):
                qualities_vector.vs[j] = partial_global_qualities.vs[i][j]

            try:
                # Use shared computation function
                z_km = compute_z_threshold(
                    partial_global_mesh, qualities_vector, z_thresh_double
                )
            except ValueError as e:
                error_msg = (
                    f"Z_Threshold error at point ({i}, lat_ind={lat_ind}): {str(e)}"
                )
                logger.log(logging.ERROR, error_msg)
                raise ValueError(error_msg)

            f.write(
                f"{partial_global_mesh.lon[i]:.6f}\t"
                f"{partial_global_mesh.lat[i]:.6f}\t"
                f"{z_km:.6f}\n"
            )

    logger.log(logging.DEBUG, f"Wrote Z{z_threshold} data to {z_file}")


@cli.from_docstring(app)
def generate_threshold_grid(
    out_dir: Annotated[
        Path, typer.Option(file_okay=False, help="Output directory for threshold map")
    ],
    model_version: Annotated[str, typer.Option(help="Version of the model to use")],
    origin_lat: Annotated[float, typer.Option(help="Origin latitude (degrees)")],
    origin_lon: Annotated[float, typer.Option(help="Origin longitude (degrees)")],
    origin_rot: Annotated[
        float, typer.Option(help="Origin rotation angle (degrees)")
    ],
    extent_x: Annotated[
        float, typer.Option(help="Extent in X direction (kilometers)")
    ],
    extent_y: Annotated[
        float, typer.Option(help="Extent in Y direction (kilometers)")
    ],
    extent_latlon_spacing: Annotated[
        float, typer.Option(help="Spacing between lat/lon grid points (kilometers)")
    ],
    vs_type: Annotated[
        VSType, typer.Option(help="Threshold velocity type (VS30, VS500, Z1.0, Z2.5)")
    ],
    nzcvm_registry: Annotated[
        Path | None,
        typer.Option(
            exists=False,
            dir_okay=False,
            help="Path to nzcvm_registry.yaml (default: nzcvm_data/nzcvm_registry.yaml)",
        ),
    ] = None,
    nzcvm_data_root: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            exists=False,
            help="Override the default DATA_ROOT directory",
        ),
    ] = None,
    log_level: Annotated[str, typer.Option(help="Logging level")] = "INFO",
) -> None:
    """
    Generate threshold velocity map (VS30, VS500, Z1.0, or Z2.5) for a geographic region.

    This function orchestrates the generation of threshold velocity maps:
    1. Sets up the model extent and depth parameters based on VS_TYPE
    2. Generates a 2D geographic grid over the specified region
    3. Loads all required velocity model data (global models, tomography, basins)
    4. For each grid point, computes velocity profile with depth
    5. Calculates the threshold metric:
       - VS30/VS500: Time-averaged shear-wave velocity over specified depth
       - Z1.0/Z2.5: Depth where Vs first exceeds the threshold velocity
    6. Writes results to a tab-separated text file

    Parameters
    ----------
    out_dir : Path
        Path to the output directory where threshold files will be written.
    model_version : str
        Version of the velocity model to use (e.g., "2.07").
    origin_lat : float
        Latitude of the model origin (degrees).
    origin_lon : float
        Longitude of the model origin (degrees).
    origin_rot : float
        Rotation angle of the model grid (degrees, clockwise from North).
    extent_x : float
        Extent of the model in X direction (kilometers).
    extent_y : float
        Extent of the model in Y direction (kilometers).
    extent_latlon_spacing : float
        Spacing between grid points (kilometers).
    vs_type : VSType
        Type of threshold to calculate (VS30, VS500, Z1.0, or Z2.5).
    nzcvm_registry : Path, optional
        Path to the model registry file (default: nzcvm_data/nzcvm_registry.yaml).
    nzcvm_data_root : Path, optional
        Override the default nzcvm_data directory.
    log_level : str, optional
        Logging level for the script (default: "INFO").

    Raises
    ------
    ValueError
        If input parameters are invalid or threshold is outside depth limits.
    OSError
        If there are issues creating directories or writing files.
    RuntimeError
        If an error occurs during map generation or data processing.
    """
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logger.setLevel(numeric_level)

    start_time = time.time()
    logger.log(logging.INFO, f"Starting threshold grid generation: {vs_type}")
    logger.log(logging.INFO, f"Model version: {model_version}")
    logger.log(
        logging.INFO,
        f"Region: origin=({origin_lat:.4f}, {origin_lon:.4f}), "
        f"extent=({extent_x:.1f} x {extent_y:.1f} km), "
        f"rotation={origin_rot:.1f}°",
    )

    # Get data root
    data_root = get_data_root(nzcvm_data_root)
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

    # Create output directory
    out_dir = out_dir.resolve()
    try:
        out_dir.mkdir(exist_ok=True, parents=True)
        logger.log(logging.INFO, f"Output directory: {out_dir}")
    except OSError as e:
        logger.log(logging.ERROR, f"Failed to create output directory {out_dir}: {e}")
        raise OSError(f"Failed to create output directory {out_dir}: {str(e)}")

    # Get depth parameters based on VS_TYPE
    zmax, zmin, h_depth = get_depth_parameters(vs_type)
    logger.log(
        logging.INFO,
        f"Depth range for {vs_type}: {zmin:.4f} to {zmax:.4f} km, spacing {h_depth:.4f} km",
    )

    # Set up model parameters (SQUASHED topography is hardcoded as in C version)
    vm_params = {
        "model_version": model_version,
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
        "origin_rot": origin_rot,
        "extent_x": extent_x,
        "extent_y": extent_y,
        "extent_zmax": zmax,
        "extent_zmin": zmin,
        "h_depth": h_depth,
        "h_lat_lon": extent_latlon_spacing,
        "topo_type": "SQUASHED",  # Hardcoded as in C version
    }

    # Initialize registry
    logger.log(logging.INFO, "Initializing velocity model registry")
    cvm_registry = CVMRegistry(model_version, data_root, nzcvm_registry, logger)

    # Generate model grid
    logger.log(logging.INFO, "Generating model grid")
    global_mesh = gen_full_model_grid_great_circle(vm_params, logger)
    logger.log(
        logging.INFO,
        f"Grid dimensions: nx={global_mesh.nx}, ny={global_mesh.ny}, nz={global_mesh.nz}",
    )

    # Load all required data
    logger.log(logging.INFO, "Loading velocity model data")
    try:
        vm1d_data, nz_tomography_data, global_surfaces, basin_data_list = (
            cvm_registry.load_all_global_data()
        )
    except Exception as e:
        logger.log(logging.ERROR, f"Failed to load model data: {e}")
        raise RuntimeError(f"Failed to load model data: {str(e)}")

    # Preprocess basin membership
    logger.log(logging.INFO, "Preprocessing basin membership")
    in_basin_mesh = InBasinGlobalMesh(global_mesh, basin_data_list)

    # Process grid - loop over Y (latitude slices)
    logger.log(logging.INFO, f"Processing {global_mesh.ny} latitude slices")
    for j in tqdm(range(global_mesh.ny), desc="Processing slices"):
        # Extract partial mesh for this latitude slice
        partial_global_mesh = PartialGlobalMesh(global_mesh, j)

        # Initialize data structures for this slice
        partial_global_qualities = PartialGlobalQualities(
            partial_global_mesh.nx, partial_global_mesh.nz
        )

        # Loop over X (longitude points)
        for k in range(partial_global_mesh.nx):
            # Initialize per-point data structures
            in_basin_list = [InBasin() for _ in basin_data_list]

            # Get basin indices for this location
            basin_indices = [
                in_basin_mesh.basin_idx[j][k][basin_idx]
                for basin_idx, _ in enumerate(basin_data_list)
            ]

            # Mark which basins this point is in
            for basin_idx in basin_indices:
                if basin_idx >= 0:
                    in_basin_list[basin_idx].in_basin_lat_lon = True

            # Extract mesh vector for this point
            mesh_vector = MeshVector(partial_global_mesh, k)

            # Initialize surface depths structures
            partial_global_surface_depths = PartialGlobalSurfaceDepths(
                len(global_surfaces)
            )
            partial_basin_surface_depths = [
                PartialBasinSurfaceDepths(basin_data) for basin_data in basin_data_list
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
                in_basin_mesh,
                vm_params["topo_type"],
            )

            # Copy qualities to partial mesh
            for i in range(partial_global_mesh.nz):
                partial_global_qualities.rho[k][i] = qualities_vector.rho[i]
                partial_global_qualities.vp[k][i] = qualities_vector.vp[i]
                partial_global_qualities.vs[k][i] = qualities_vector.vs[i]

        # Calculate and save threshold values for this latitude slice
        if vs_type == VSType.VS500:
            calc_and_save_vs(
                out_dir, partial_global_mesh, partial_global_qualities, "500", j, logger
            )
        elif vs_type == VSType.VS30:
            calc_and_save_vs(
                out_dir, partial_global_mesh, partial_global_qualities, "30", j, logger
            )
        elif vs_type == VSType.Z1_0:
            calc_and_save_z_threshold(
                out_dir, partial_global_mesh, partial_global_qualities, "1.0", j, logger
            )
        elif vs_type == VSType.Z2_5:
            calc_and_save_z_threshold(
                out_dir, partial_global_mesh, partial_global_qualities, "2.5", j, logger
            )

    logger.log(logging.INFO, "Threshold grid generation 100% complete")
    logger.log(
        logging.INFO,
        f"Threshold map (type: {vs_type}, version: {model_version}) "
        f"successfully generated and written to {out_dir}",
    )
    elapsed_time = time.time() - start_time
    logger.log(
        logging.INFO,
        f"Threshold grid generation completed in {elapsed_time:.2f} seconds",
    )


if __name__ == "__main__":
    app()