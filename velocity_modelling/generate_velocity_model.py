# import concurrent.futures

from logging import Logger
import logging

from pathlib import Path
import argparse

import yaml
import numpy as np

from typing import Any

from cvm_registry import (
    CVMRegistry,
    nzvm_registry_path,
    GlobalMesh,
    ModelExtent,
    PartialGlobalMesh,
)

LON_GRID_DIM_MAX = 10260
LAT_GRID_DIM_MAX = 19010
DEP_GRID_DIM_MAX = 4500

# constants for coord generation functions
FLAT_CONST = 298.256
ERAD = 6378.139  # Earth's radius in km
RPERD = 0.017453292

from typing import Dict


def write_velo_mod_corners_text_file(
    global_mesh: GlobalMesh, output_dir: str, logger: Logger
) -> None:
    """
    Write velocity model corners to a text file.

    Parameters
    ----------
    global_mesh : GlobalMesh
        An object containing the global mesh data, including longitude and latitude arrays.
    output_dir : str
        Directory where the output log file will be saved.
    logger : Logger
        Logger for logging information.

    Returns
    -------
    None
    """
    log_file_name = Path(output_dir) / "Log" / "VeloModCorners.txt"
    log_file_name.parent.mkdir(parents=True, exist_ok=True)

    with log_file_name.open("w") as fp:
        fp.write(">Velocity model corners.\n")
        fp.write(">Lon\tLat\n")
        fp.write(
            f"{global_mesh.Lon[0][global_mesh.nY - 1]}\t{global_mesh.Lat[0][global_mesh.nY - 1]}\n"
        )
        fp.write(f"{global_mesh.Lon[0][0]}\t{global_mesh.Lat[0][0]}\n")
        fp.write(
            f"{global_mesh.Lon[global_mesh.nX - 1][0]}\t{global_mesh.Lat[global_mesh.nX - 1][0]}\n"
        )
        fp.write(
            f"{global_mesh.Lon[global_mesh.nX - 1][global_mesh.nY - 1]}\t{global_mesh.Lat[global_mesh.nX - 1][global_mesh.nY - 1]}\n"
        )

    logger.info("Velocity model corners file write complete.")


def great_circle_projection(
    x: np.ndarray,
    y: np.ndarray,
    amat: np.ndarray,
    erad: float = ERAD,
    g0: float = 0,
    b0: float = 0,
) -> tuple[np.ndarray, Any]:
    """
    Project x, y coordinates to geographic coordinates (longitude, latitude) using a great circle projection.

    Parameters
    ----------
    x : np.ndarray
        X-coordinates.
    y : np.ndarray
        Y-coordinates.
    amat : np.ndarray
        Transformation matrix.
    erad : float, optional
        Earth's radius (default is ERAD).
    g0 : float, optional
        Initial longitude (default is 0).
    b0 : float, optional
        Initial latitude (default is 0).

    Returns
    -------
    tuple[np.ndarray, Any]
        Computed latitude and longitude arrays.
    """
    cosB = np.cos(x / erad - b0)
    sinB = np.sin(x / erad - b0)

    cosG = np.cos(y / erad - g0)
    sinG = np.sin(y / erad - g0)

    xp = sinG * cosB * np.sqrt(1 + sinB * sinB * sinG * sinG)
    yp = sinB * cosG * np.sqrt(1 + sinB * sinB * sinG * sinG)
    zp = np.sqrt(1 - xp * xp - yp * yp)
    coords = np.stack((xp, yp, zp), axis=0)

    xg, yg, zg = np.tensordot(amat, coords, axes=([1], [0]))

    lat = np.where(
        np.isclose(zg, 0),
        0,
        90 - np.arctan(np.sqrt(xg**2 + yg**2) / zg) / RPERD - np.where(zg < 0, 180, 0),
    )

    lon = np.where(
        np.isclose(xg, 0), 0, np.arctan(yg / xg) / RPERD - np.where(xg < 0, 180, 0)
    )
    lon = lon % 360
    return lat, lon


def gen_full_model_grid_great_circle(
    model_extent: ModelExtent, logger: Logger
) -> GlobalMesh:
    """
    Generate the grid of latitude, longitude, and depth points using the point radial distance method.

    Parameters
    ----------
    model_extent : ModelExtent
        Object containing the extent, spacing, and version of the model.
    logger : Logger
        Logger for logging information.

    Returns
    -------
    GlobalMesh
        Object containing the generated grid of latitude, longitude, and depth points.
    """
    nX = int(np.round(model_extent.Xmax / model_extent.hLatLon))
    nY = int(np.round(model_extent.Ymax / model_extent.hLatLon))
    nZ = int(np.round((model_extent.Zmax - model_extent.Zmin) / model_extent.hDep))

    global_mesh = GlobalMesh(nX, nY, nZ)

    global_mesh.maxLat = -180
    global_mesh.minLat = 0
    global_mesh.maxLon = 0
    global_mesh.minLon = 180

    assert global_mesh.nX == model_extent.nx
    assert global_mesh.nY == model_extent.ny
    assert global_mesh.nZ == model_extent.nz

    if any(
        [
            global_mesh.nX >= LON_GRID_DIM_MAX,
            global_mesh.nY >= LAT_GRID_DIM_MAX,
            global_mesh.nZ >= DEP_GRID_DIM_MAX,
        ]
    ):
        raise ValueError(
            f"Grid dimensions exceed maximum allowable values. X={LON_GRID_DIM_MAX}, Y={LAT_GRID_DIM_MAX}, Z={DEP_GRID_DIM_MAX}"
        )

    if global_mesh.nZ != 1:
        logger.info(
            f"Number of model points. nx: {global_mesh.nX}, ny: {global_mesh.nY}, nz: {global_mesh.nZ}."
        )

    for i in range(global_mesh.nX):
        global_mesh.X[i] = (
            0.5 * model_extent.hLatLon
            + model_extent.hLatLon * i
            - 0.5 * model_extent.Xmax
        )

    for i in range(global_mesh.nY):
        global_mesh.Y[i] = (
            0.5 * model_extent.hLatLon
            + model_extent.hLatLon * i
            - 0.5 * model_extent.Ymax
        )

    for i in range(global_mesh.nZ):
        global_mesh.Z[i] = -1000 * (model_extent.Zmin + model_extent.hDep * (i + 0.5))

    arg = model_extent.originRot * RPERD
    cosA = np.cos(arg)
    sinA = np.sin(arg)

    arg = (90.0 - model_extent.originLat) * RPERD
    cosT = np.cos(arg)
    sinT = np.sin(arg)

    arg = model_extent.originLon * RPERD
    cosP = np.cos(arg)
    sinP = np.sin(arg)

    amat = np.array(
        [
            cosA * cosT * cosP + sinA * sinP,
            sinA * cosT * cosP - cosA * sinP,
            sinT * cosP,
            cosA * cosT * sinP - sinA * cosP,
            sinA * cosT * sinP + cosA * cosP,
            sinT * sinP,
            -cosA * sinT,
            -sinA * sinT,
            cosT,
        ]
    ).reshape((3, 3))

    det = np.linalg.det(amat)
    ainv = np.linalg.inv(amat) / det

    g0 = 0.0
    b0 = 0.0

    X, Y = np.meshgrid(
        global_mesh.X[: global_mesh.nX], global_mesh.Y[: global_mesh.nY], indexing="ij"
    )
    lat_lon = great_circle_projection(X, Y, amat, ERAD, g0, b0)
    (
        global_mesh.Lat[: global_mesh.nX, : global_mesh.nY],
        global_mesh.Lon[: global_mesh.nX, : global_mesh.nY],
    ) = lat_lon

    global_mesh.maxLat = np.max(global_mesh.Lat)
    global_mesh.maxLon = np.max(global_mesh.Lon)
    global_mesh.minLat = np.min(global_mesh.Lat)
    global_mesh.minLon = np.min(global_mesh.Lon)

    logger.info("Completed Generation of Model Grid.")
    return global_mesh


def extract_partial_mesh(global_mesh: GlobalMesh, lat_ind: int) -> PartialGlobalMesh:
    """
    Extract one slice of values from the global mesh, i.e., nX x nY x nZ becomes nX x 1 x nZ.

    Parameters
    ----------
    global_mesh : GlobalMesh
        The global mesh containing the full model grid (lat, lon, and depth points).
    lat_ind : int
        The y index of the slice of the global grid to be extracted.

    Returns
    -------
    PartialGlobalMesh
        A struct containing a slice of the global mesh.
    """
    partial_global_mesh = PartialGlobalMesh(global_mesh.nX, global_mesh.nZ)
    partial_global_mesh.Y = global_mesh.Y[lat_ind]

    partial_global_mesh.Z = global_mesh.Z.copy()
    partial_global_mesh.Lon = global_mesh.Lon[:, lat_ind].copy()
    partial_global_mesh.Lat = global_mesh.Lat[:, lat_ind].copy()
    partial_global_mesh.X = global_mesh.X.copy()

    return partial_global_mesh


def generate_velocity_model(
    cvm_registry: CVMRegistry, out_dir: Path, vm_params: Dict, logger: Logger
):
    """
    Generate the velocity model.

    Parameters
    ----------
    cvm_registry : CVMRegistry
        The CVMRegistry instance.
    out_dir : Path
        The output directory path.
    vm_params : Dict
        The velocity model parameters.
    logger : Logger
        Logger for logging information.
    """
    # Implementation of the function
    model_extent = ModelExtent(vm_params)
    global_mesh = gen_full_model_grid_great_circle(model_extent, logger)
    write_velo_mod_corners_text_file(global_mesh, out_dir, logger)

    velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data = (
        cvm_registry.load_all_global_data(logger)
    )

    for j in range(global_mesh.nY):
        logger.info(
            f"Generating velocity model {j * 100 / global_mesh.nY:.2f}% complete."
        )
        partial_global_mesh = extract_partial_mesh(global_mesh, j)
        # partial_global_qualities = PartialGlobalQualities(partial_global_mesh.nX, partial_global_mesh.nZ)
    #
    #     for k in range(partial_global_mesh.nX):
    #         in_basin = InBasin()
    #         partial_global_surface_depths = PartialGlobalSurfaceDepths()
    #         partial_basin_surface_depths = PartialBasinSurfaceDepths()
    #         qualities_vector = QualitiesVector()
    #         extended_qualities_vector = QualitiesVector()
    #
    #         if smoothing_required == 1:
    #             extended_mesh_vector = extend_mesh_vector(partial_global_mesh, n_pts_smooth, model_extent.hDep * 1000,
    #                                                       k)
    #             assign_qualities(global_model_parameters, velo_mod_1d_data, nz_tomography_data, global_surfaces,
    #                              basin_data, extended_mesh_vector, partial_global_surface_depths,
    #                              partial_basin_surface_depths, in_basin, extended_qualities_vector, logger,
    #                              gen_extract_velo_mod_call.topo_type)
    #
    #             for i in range(partial_global_mesh.nZ):
    #                 mid_pt_count = i * (1 + 2 * n_pts_smooth) + 1
    #                 mid_pt_count_plus = mid_pt_count + 1
    #                 mid_pt_count_minus = mid_pt_count - 1
    #
    #                 A = one_third * extended_qualities_vector.Rho[mid_pt_count_minus]
    #                 B = four_thirds * extended_qualities_vector.Rho[mid_pt_count]
    #                 C = one_third * extended_qualities_vector.Rho[mid_pt_count_plus]
    #                 partial_global_qualities.Rho[k][i] = half * (A + B + C)
    #
    #                 A = one_third * extended_qualities_vector.Vp[mid_pt_count_minus]
    #                 B = four_thirds * extended_qualities_vector.Vp[mid_pt_count]
    #                 C = one_third * extended_qualities_vector.Vp[mid_pt_count_plus]
    #                 partial_global_qualities.Vp[k][i] = half * (A + B + C)
    #
    #                 A = one_third * extended_qualities_vector.Vs[mid_pt_count_minus]
    #                 B = four_thirds * extended_qualities_vector.Vs[mid_pt_count]
    #                 C = one_third * extended_qualities_vector.Vs[mid_pt_count_plus]
    #                 partial_global_qualities.Vs[k][i] = half * (A + B + C)
    #         else:
    #             mesh_vector = extract_mesh_vector(partial_global_mesh, k)
    #             assign_qualities(global_model_parameters, velo_mod_1d_data, nz_tomography_data, global_surfaces,
    #                              basin_data, mesh_vector, partial_global_surface_depths, partial_basin_surface_depths,
    #                              in_basin, qualities_vector, calculation_log, gen_extract_velo_mod_call.topo_type)
    #
    #             for i in range(partial_global_mesh.nZ):
    #                 partial_global_qualities.Rho[k][i] = qualities_vector.Rho[i]
    #                 partial_global_qualities.Vp[k][i] = qualities_vector.Vp[i]
    #                 partial_global_qualities.Vs[k][i] = qualities_vector.Vs[i]
    #                 partial_global_qualities.inbasin[k][i] = qualities_vector.inbasin[i]
    #
    #     write_global_qualities(output_dir, partial_global_mesh, partial_global_qualities, gen_extract_velo_mod_call,
    #                            calculation_log, j)

    # def process_j(j):
    #     # do something with j
    #
    #     return j
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_j, j) for j in range(global_mesh.nY)]
    #     for future in concurrent.futures.as_completed(futures):
    #         j = future.result()
    #          logger.info(f"Generating velocity model {j * 100 / global_mesh.nY:.2f}% complete.", end="")
    #         sys.stdout.flush()

    logger.info("Generation of velocity model 100% complete.")
    logger.info("Model generation complete.")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate velocity model")
    parser.add_argument("vm_params", type=Path, help="Path to the vm_params.yaml file")
    parser.add_argument("out_dir", type=Path, help="Path to the output directory")
    parser.add_argument(
        "--nzvm_registry",
        type=Path,
        help="Path to the nzvm_registry.yaml file",
        default=nzvm_registry_path,
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    vm_params_path = args.vm_params
    assert vm_params_path.exists(), f"File is not present: {vm_params_path}"
    assert args.nzvm_registry.exists(), f"File is not present: {args.nzvm_registry}"

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Using vm_params file: {vm_params_path}")
    with open(vm_params_path, "r") as f:
        vm_params = yaml.safe_load(f)

    cvm_registry = CVMRegistry(vm_params["model_version"], logger)

    generate_velocity_model(cvm_registry, out_dir, vm_params, logger)
