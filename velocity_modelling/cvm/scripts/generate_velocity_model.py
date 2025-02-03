# import concurrent.futures

from logging import Logger
import logging

from pathlib import Path
import argparse

import yaml
import numpy as np

from typing import Any, List, Dict

from velocity_modelling.cvm.registry import (
    nzvm_registry_path,
    CVMRegistry,
    GlobalMesh,
    MeshVector,
    ModelExtent,
    PartialGlobalMesh,
    PartialGlobalQualities,
    VeloMod1DData,
    TomographyData,
    GlobalSurfaces,
    BasinData,
    InBasin,
    PartialGlobalSurfaceDepths,
    PartialBasinSurfaceDepths,
    QualitiesVector,
)

from velocity_modelling.cvm.constants import MAX_DIST_SMOOTH

LON_GRID_DIM_MAX = 10260
LAT_GRID_DIM_MAX = 19010
DEP_GRID_DIM_MAX = 4500

# constants for coord generation functions
FLAT_CONST = 298.256
ERAD = 6378.139  # Earth's radius in km
RPERD = 0.017453292


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

    nx = len(global_mesh.x)
    ny = len(global_mesh.y)

    with log_file_name.open("w") as fp:
        fp.write(">Velocity model corners.\n")
        fp.write(">Lon\tLat\n")
        fp.write(f"{global_mesh.lon[0][ny - 1]}\t{global_mesh.lat[0][ny - 1]}\n")
        fp.write(f"{global_mesh.lon[0][0]}\t{global_mesh.lat[0][0]}\n")
        fp.write(f"{global_mesh.lon[nx - 1][0]}\t{global_mesh.lat[nx - 1][0]}\n")
        fp.write(
            f"{global_mesh.lon[nx - 1][ny - 1]}\t{global_mesh.lat[nx - 1][ny - 1]}\n"
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
    nx = int(np.round(model_extent.xmax / model_extent.h_lat_lon))
    ny = int(np.round(model_extent.ymax / model_extent.h_lat_lon))
    nz = int(np.round((model_extent.zmax - model_extent.zmin) / model_extent.h_depth))

    global_mesh = GlobalMesh(nx, ny, nz)

    global_mesh.max_lat = -180
    global_mesh.min_lat = 0
    global_mesh.max_lon = 0
    global_mesh.min_lon = 180

    assert nx == model_extent.nx
    assert ny == model_extent.ny
    assert nz == model_extent.nz

    if any(
        [
            nx >= LON_GRID_DIM_MAX,
            ny >= LAT_GRID_DIM_MAX,
            nz >= DEP_GRID_DIM_MAX,
        ]
    ):
        raise ValueError(
            f"Grid dimensions exceed maximum allowable values. X={LON_GRID_DIM_MAX}, Y={LAT_GRID_DIM_MAX}, Z={DEP_GRID_DIM_MAX}"
        )

    if nz != 1:
        logger.info(f"Number of model points. nx: {nx}, ny: {ny}, nz: {nz}.")

    for i in range(nx):
        global_mesh.x[i] = (
            0.5 * model_extent.h_lat_lon
            + model_extent.h_lat_lon * i
            - 0.5 * model_extent.xmax
        )

    for i in range(ny):
        global_mesh.y[i] = (
            0.5 * model_extent.h_lat_lon
            + model_extent.h_lat_lon * i
            - 0.5 * model_extent.ymax
        )

    for i in range(nz):
        global_mesh.z[i] = -1000 * (
            model_extent.zmin + model_extent.h_depth * (i + 0.5)
        )

    arg = model_extent.origin_rot * RPERD
    cosA = np.cos(arg)
    sinA = np.sin(arg)

    arg = (90.0 - model_extent.origin_lat) * RPERD
    cosT = np.cos(arg)
    sinT = np.sin(arg)

    arg = model_extent.origin_lon * RPERD
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

    x, y = np.meshgrid(global_mesh.x[:nx], global_mesh.y[:ny], indexing="ij")
    lat_lon = great_circle_projection(x, y, amat, ERAD, g0, b0)
    (
        global_mesh.lat[:nx, :ny],
        global_mesh.lon[:nx, :ny],
    ) = lat_lon

    global_mesh.max_lat = np.max(global_mesh.lat)
    global_mesh.max_lon = np.max(global_mesh.lon)
    global_mesh.min_lat = np.min(global_mesh.lat)
    global_mesh.min_lon = np.min(global_mesh.lon)

    logger.info("Completed Generation of Model Grid.")
    return global_mesh


def extract_partial_mesh(global_mesh: GlobalMesh, lat_ind: int) -> PartialGlobalMesh:
    """
    Extract one slice of values from the global mesh, i.e., nx x ny x nz becomes nx x 1 x nz.

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
    partial_global_mesh = PartialGlobalMesh(global_mesh.nx, global_mesh.nz)
    partial_global_mesh.y = global_mesh.y[lat_ind]

    partial_global_mesh.z = global_mesh.z.copy()
    partial_global_mesh.lon = global_mesh.lon[:, lat_ind].copy()
    partial_global_mesh.lat = global_mesh.lat[:, lat_ind].copy()
    partial_global_mesh.x = global_mesh.x.copy()

    return partial_global_mesh


def extract_mesh_vector(partial_global_mesh: PartialGlobalMesh, lon_ind: int):
    """
    Extract one vector of values from the global mesh, i.e., nx x 1 x nz becomes 1 x 1 x nz.

    Parameters
    ----------
    partial_global_mesh : PartialGlobalMesh
        The partial global mesh containing the slice of the global grid.
    lon_ind : int
        The x index of the slice of the grid to be extracted.

    Returns
    -------
    MeshVector
        A struct containing one lat-lon point and the depths of all grid points at this location.
    """
    mesh_vector = MeshVector(len(partial_global_mesh.z))

    mesh_vector.lat = partial_global_mesh.lat[lon_ind]
    mesh_vector.lon = partial_global_mesh.lon[lon_ind]
    mesh_vector.z = partial_global_mesh.z.copy()

    return mesh_vector


def assign_qualities(
    cvm_registry: CVMRegistry,
    velo_mod_1d_data: VeloMod1DData,
    nz_tomography_data: TomographyData,
    global_surfaces: GlobalSurfaces,
    basin_data_list: List[BasinData],
    mesh_vector: MeshVector,
    partial_global_surface_depths: PartialGlobalSurfaceDepths,
    partial_basin_surface_depths_list: List[PartialBasinSurfaceDepths],
    in_basin_list: List[InBasin],
    qualities_vector: QualitiesVector,
    topo_type: str,
    logger: Logger,
):
    """
    Determine if lat-lon point lies within the smoothing zone and prescribe velocities accordingly.
    """
    smooth_bound = nz_tomography_data.smooth_boundary

    closest_ind = None

    if smooth_bound.n == 0:
        distance = 1e6  # if there are no points in the smoothing boundary, then skip
    else:
        closest_ind, distance = (
            smooth_bound.determine_if_lat_lon_within_smoothing_region(mesh_vector)
        )

    # calculate vs30 (used as a proxy to determine if point is on- or off-shore, only if using tomography)
    if nz_tomography_data.tomography_loaded and cvm_registry.vm_global_params["GTL"]:
        nz_tomography_data.calculate_vs30_from_tomo_vs30_surface(
            mesh_vector
        )  # mesh_vector.vs30 updated
        nz_tomography_data.calculate_distance_from_shoreline(
            mesh_vector
        )  # mesh_vector.distance_from_shoreline updated

    in_any_basin = np.any(
        [
            basin_data.determine_if_within_basin_lat_lon(mesh_vector)
            for basin_data in basin_data_list
        ]
    )

    # point lies within smoothing zone, is offshore, and is not in any basin (i.e., outside any boundaries)
    if (
        distance <= MAX_DIST_SMOOTH
        and not in_any_basin
        and cvm_registry.vm_global_params["GTL"]
        and mesh_vector.vs30 < 100
    ):
        # point lies within smoothing zone and is not in any basin (i.e., outside any boundaries)
        qualities_vector_a = QualitiesVector(mesh_vector.nz)
        qualities_vector_b = QualitiesVector(mesh_vector.nz)
        in_basin_b_list = [
            InBasin(basin_data, mesh_vector.nz) for basin_data in basin_data_list
        ]
        partial_global_surface_depths_b = PartialGlobalSurfaceDepths(mesh_vector.nz)
        partial_basin_surface_depths_list_b = [
            PartialBasinSurfaceDepths(mesh_vector.nz) for basin_data in basin_data_list
        ]

        original_lat = mesh_vector.lat
        original_lon = mesh_vector.lon

        # overwrite the lat-lon with the location on the boundary
        assert (
            closest_ind is not None
        )  # closest_ind should not be None if distance < MAX_DIST_SMOOTH
        mesh_vector.lat = smooth_bound.yPts[closest_ind]
        mesh_vector.lon = smooth_bound.xPts[closest_ind]

        # velocity vector just inside the boundary
        on_boundary = True
        qualities_vector_b.prescribe_velocities(
            velo_mod_1d_data,
            nz_tomography_data,
            global_surfaces,
            basin_data_list,
            mesh_vector,
            partial_global_surface_depths_b,
            partial_basin_surface_depths_list_b,
            in_basin_b_list,
            topo_type,
            on_boundary,
            logger,
        )

        # overwrite the lat-lon with the original lat-lon point
        mesh_vector.lat = original_lat
        mesh_vector.lon = original_lon

        # velocity vector at the point in question
        on_boundary = False
        qualities_vector_a.prescribe_velocities(
            velo_mod_1d_data,
            nz_tomography_data,
            global_surfaces,
            basin_data_list,
            mesh_vector,
            partial_global_surface_depths,
            partial_basin_surface_depths_list,
            in_basin_list,
            topo_type,
            on_boundary,
            logger,
        )

        # apply smoothing between the two generated velocity vectors
        smooth_dist_ratio = distance / MAX_DIST_SMOOTH
        inverse_ratio = 1 - smooth_dist_ratio

        valid_indices = ~np.isnan(qualities_vector_a.vp)
        qualities_vector.vp[valid_indices] = (
            smooth_dist_ratio * qualities_vector_a.vp[valid_indices]
            + inverse_ratio * qualities_vector_b.vp[valid_indices]
        )
        qualities_vector.vs[valid_indices] = (
            smooth_dist_ratio * qualities_vector_a.vs[valid_indices]
            + inverse_ratio * qualities_vector_b.vs[valid_indices]
        )
        qualities_vector.rho[valid_indices] = (
            smooth_dist_ratio * qualities_vector_a.rho[valid_indices]
            + inverse_ratio * qualities_vector_b.rho[valid_indices]
        )

        invalid_indices = np.isnan(qualities_vector_a.vp)
        qualities_vector.vp[invalid_indices] = np.nan
        qualities_vector.vs[invalid_indices] = np.nan
        qualities_vector.rho[invalid_indices] = np.nan
    else:
        on_boundary = False
        qualities_vector.prescribe_velocities(
            velo_mod_1d_data,
            nz_tomography_data,
            global_surfaces,
            basin_data_list,
            mesh_vector,
            partial_global_surface_depths,
            partial_basin_surface_depths_list,
            in_basin_list,
            topo_type,
            on_boundary,
            logger,
        )


def generate_velocity_model(
    cvm_registry: CVMRegistry,
    out_dir: Path,
    vm_params: Dict,
    logger: Logger,
    smoothing: bool = False,
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
    smoothing : bool, optional
        Whether to apply smoothing to the model (default is False).
    """
    # Implementation of the function
    model_extent = ModelExtent(vm_params)
    global_mesh = gen_full_model_grid_great_circle(model_extent, logger)
    write_velo_mod_corners_text_file(global_mesh, out_dir, logger)

    velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data_list = (
        cvm_registry.load_all_global_data(logger)
    )

    for j in range(len(global_mesh.y)):
        logger.info(
            f"Generating velocity model {j * 100 / len(global_mesh.y):.2f}% complete."
        )
        partial_global_mesh = extract_partial_mesh(global_mesh, j)
        partial_global_qualities = PartialGlobalQualities(
            partial_global_mesh.nx, partial_global_mesh.nz
        )

        for k in range(len(partial_global_mesh.x)):
            in_basin_list = [
                InBasin(basin_data, partial_global_mesh.nz)
                for basin_data in basin_data_list
            ]
            partial_global_surface_depths = PartialGlobalSurfaceDepths(
                partial_global_mesh.nz
            )

            partial_basin_surface_depths_list = [
                PartialBasinSurfaceDepths(basin_data) for basin_data in basin_data_list
            ]
            qualities_vector = QualitiesVector(partial_global_mesh.nz)
            extended_qualities_vector = QualitiesVector(partial_global_mesh.nz)
            #
            if smoothing:
                pass
            #             extended_mesh_vector = extend_mesh_vector(partial_global_mesh, n_pts_smooth, model_extent.hDep * 1000,
            #                                                       k)
            #             assign_qualities(global_model_parameters, velo_mod_1d_data, nz_tomography_data, global_surfaces,
            #                              basin_data, extended_mesh_vector, partial_global_surface_depths,
            #                              partial_basin_surface_depths, in_basin, extended_qualities_vector, logger,
            #                              gen_extract_velo_mod_call.topo_type)
            #
            #             for i in range(partial_global_mesh.nz):
            #                 mid_pt_count = i * (1 + 2 * n_pts_smooth) + 1
            #                 mid_pt_count_plus = mid_pt_count + 1
            #                 mid_pt_count_minus = mid_pt_count - 1
            #
            #                 A = one_third * extended_qualities_vector.rho[mid_pt_count_minus]
            #                 B = four_thirds * extended_qualities_vector.rho[mid_pt_count]
            #                 C = one_third * extended_qualities_vector.rho[mid_pt_count_plus]
            #                 partial_global_qualities.rho[k][i] = half * (A + B + C)
            #
            #                 A = one_third * extended_qualities_vector.vp[mid_pt_count_minus]
            #                 B = four_thirds * extended_qualities_vector.vp[mid_pt_count]
            #                 C = one_third * extended_qualities_vector.vp[mid_pt_count_plus]
            #                 partial_global_qualities.vp[k][i] = half * (A + B + C)
            #
            #                 A = one_third * extended_qualities_vector.vs[mid_pt_count_minus]
            #                 B = four_thirds * extended_qualities_vector.vs[mid_pt_count]
            #                 C = one_third * extended_qualities_vector.vs[mid_pt_count_plus]
            #                 partial_global_qualities.vs[k][i] = half * (A + B + C)
            else:
                mesh_vector = extract_mesh_vector(partial_global_mesh, k)
                assign_qualities(
                    cvm_registry,
                    velo_mod_1d_data,
                    nz_tomography_data,
                    global_surfaces,
                    basin_data_list,
                    mesh_vector,
                    partial_global_surface_depths,
                    partial_basin_surface_depths_list,
                    in_basin_list,
                    qualities_vector,
                    vm_params["topo_type"],
                    logger,
                )
                nz = partial_global_mesh.nz
                partial_global_qualities.rho[k, :nz] = qualities_vector.rho[:nz]
                partial_global_qualities.vp[k, :nz] = qualities_vector.vp[:nz]
                partial_global_qualities.vs[k, :nz] = qualities_vector.vs[:nz]
                partial_global_qualities.inbasin[k, :nz] = qualities_vector.inbasin[:nz]

        # write_global_qualities(
        #     output_dir,
        #     partial_global_mesh,
        #     partial_global_qualities,
        #     gen_extract_velo_mod_call,
        #     calculation_log,
        #     j,
        # )

    # def process_j(j):
    #     # do something with j
    #
    #     return j
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_j, j) for j in range(global_mesh.ny)]
    #     for future in concurrent.futures.as_completed(futures):
    #         j = future.result()
    #          logger.info(f"Generating velocity model {j * 100 / global_mesh.ny:.2f}% complete.", end="")
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
