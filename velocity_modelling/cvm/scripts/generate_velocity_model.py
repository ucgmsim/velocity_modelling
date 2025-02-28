# import concurrent.futures
from copy import deepcopy
from logging import Logger
import logging

from pathlib import Path
import argparse
import numpy as np

import yaml

from typing import Dict, List

from velocity_modelling.cvm.constants import NZVM_REGISTRY_PATH
from velocity_modelling.cvm.basin_model import (
    InBasin,
    PartialBasinSurfaceDepths,
    BasinData,
    InBasinGlobalMesh,
    preprocess_basin_membership,
)
from velocity_modelling.cvm.geometry import (
    gen_full_model_grid_great_circle,
    extract_partial_mesh,
    extract_mesh_vector,
    GlobalMesh,
    ModelExtent,
)
from velocity_modelling.cvm.global_model import PartialGlobalSurfaceDepths
from velocity_modelling.cvm.registry import CVMRegistry
from velocity_modelling.cvm.velocity3d import PartialGlobalQualities, QualitiesVector
from velocity_modelling.cvm.write.emod3d import write_global_qualities


# constants for coord generation functions


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
        Directory where the write log file will be saved.
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


# def preprocess_basin_membership(
#     global_mesh: GlobalMesh, basin_data_list: List[BasinData]
# ) -> List[InBasin]:
#     """
#     Preprocess basin membership and return a list of InBasin objects with updated in_basin_lat_lon.
#
#     Parameters
#     ----------
#     global_mesh : GlobalMesh
#         The global mesh containing lat/lon values.
#     basin_data_list : List[BasinData]
#         List of BasinData objects for each basin.
#
#     Returns
#     -------
#     List[InBasin]
#         List of InBasin objects with updated in_basin_lat_lon attributes.
#     """
#     # Initialize list of InBasin objects
#     in_basin_list = [InBasin(basin_data, len(global_mesh.z)) for basin_data in basin_data_list]
#
#     # Update membership for each basin
#     for in_basin in in_basin_list:
#         in_basin.update_basin_membership(global_mesh)
#
#     return in_basin_list


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
        The write directory path.
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

    in_basin_mesh, partial_global_mesh_list = preprocess_basin_membership(
        global_mesh, basin_data_list
    )

    for j in range(len(global_mesh.y)):
        logger.info(
            f"Generating velocity model {j * 100 / len(global_mesh.y):.2f}% complete."
        )
        partial_global_mesh = partial_global_mesh_list[
            j
        ]  # extract_partial_mesh(global_mesh, j)
        partial_global_qualities = PartialGlobalQualities(
            partial_global_mesh.nx, partial_global_mesh.nz
        )

        for k in range(len(partial_global_mesh.x)):

            partial_global_surface_depths = PartialGlobalSurfaceDepths(
                len(global_surfaces.surfaces)
            )

            partial_basin_surface_depths_list = [
                PartialBasinSurfaceDepths(basin_data) for basin_data in basin_data_list
            ]
            qualities_vector = QualitiesVector(partial_global_mesh.nz)
            extended_qualities_vector = QualitiesVector(partial_global_mesh.nz)
            #

            basin_idx = in_basin_mesh.basin_membership[j, k]
            in_basin_list = [
                InBasin(basin_data, len(global_mesh.z))
                for basin_data in basin_data_list
            ]
            if basin_idx >= 0:  # -1 means not in any basin
                in_basin_list[basin_idx].in_basin_lat_lon = True

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
                qualities_vector.assign_qualities(
                    cvm_registry,
                    velo_mod_1d_data,
                    nz_tomography_data,
                    global_surfaces,
                    basin_data_list,
                    mesh_vector,
                    partial_global_surface_depths,
                    partial_basin_surface_depths_list,
                    in_basin_list,
                    vm_params["topo_type"],
                    logger,
                )
                nz = partial_global_mesh.nz
                # partial_global_qualities.rho[k, :nz] = qualities_vector.rho[:nz]
                # partial_global_qualities.vp[k, :nz] = qualities_vector.vp[:nz]
                # partial_global_qualities.vs[k, :nz] = qualities_vector.vs[:nz]
                # partial_global_qualities.inbasin[k, :nz] = qualities_vector.inbasin[:nz]
                partial_global_qualities.rho[k] = qualities_vector.rho
                partial_global_qualities.vp[k] = qualities_vector.vp
                partial_global_qualities.vs[k] = qualities_vector.vs
                partial_global_qualities.inbasin[k] = qualities_vector.inbasin

        write_global_qualities(
            out_dir,
            partial_global_mesh,
            partial_global_qualities,
            vm_params,
            j,
            logger,
        )

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
    parser.add_argument("out_dir", type=Path, help="Path to the write directory")
    parser.add_argument(
        "--nzvm_registry",
        type=Path,
        help="Path to the nzvm_registry.yaml file",
        default=NZVM_REGISTRY_PATH,
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

    cvm_registry = CVMRegistry(
        vm_params["model_version"], args.nzvm_registry, logger=logger
    )

    generate_velocity_model(cvm_registry, out_dir, vm_params, logger)
