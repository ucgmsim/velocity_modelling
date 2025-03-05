# python

"""
This module provides functionality for generating velocity models, writing
velocity model corners to a text file, and an empty command for testing.

.. module:: nzcvm
"""

from logging import Logger
import logging
from pathlib import Path
import yaml
from typing import Annotated, Dict

import typer
from qcore import cli

from velocity_modelling.cvm.constants import NZVM_REGISTRY_PATH
from velocity_modelling.cvm.basin_model import (
    InBasin,
    PartialBasinSurfaceDepths,
    InBasinGlobalMesh,
)
from velocity_modelling.cvm.geometry import (
    gen_full_model_grid_great_circle,
    extract_mesh_vector,
    GlobalMesh,
    ModelExtent,
)
from velocity_modelling.cvm.global_model import PartialGlobalSurfaceDepths
from velocity_modelling.cvm.registry import CVMRegistry
from velocity_modelling.cvm.velocity3d import PartialGlobalQualities, QualitiesVector
from velocity_modelling.cvm.write.emod3d import write_global_qualities

app = typer.Typer(pretty_exceptions_enable=False)


def write_velo_mod_corners_text_file(
    global_mesh: GlobalMesh, output_dir: str, logger: Logger
) -> None:
    """
    Write velocity model corners to a text file.

    Parameters
    ----------
    global_mesh : GlobalMesh
        Global mesh containing the longitude and latitude arrays.
    output_dir : str
        Directory for writing the output text file.
    logger : Logger
        Logger instance for status reporting.

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
        fp.write(">Lon\\tLat\n")
        fp.write(f"{global_mesh.lon[0][ny - 1]}\\t{global_mesh.lat[0][ny - 1]}\n")
        fp.write(f"{global_mesh.lon[0][0]}\\t{global_mesh.lat[0][0]}\n")
        fp.write(f"{global_mesh.lon[nx - 1][0]}\\t{global_mesh.lat[nx - 1][0]}\n")
        fp.write(
            f"{global_mesh.lon[nx - 1][ny - 1]}\\t{global_mesh.lat[nx - 1][ny - 1]}\n"
        )

    logger.info("Velocity model corners file write complete.")


@cli.from_docstring(app)
def generate_velocity_model(
    vm_params_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out_dir: Annotated[Path, typer.Argument(file_okay=False)],
    nzvm_registry: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = NZVM_REGISTRY_PATH,
    smoothing: Annotated[bool, typer.Option()] = False,
) -> None:
    """
    Generate a velocity model.

    Parameters
    ----------
    vm_params_path : Path
        Path to the `vm_params.yaml` file containing velocity model parameters.
    out_dir : Path
        Directory where velocity model files will be written.
    nzvm_registry : Path, optional
        Path to the `nzvm_registry.yaml` file. Defaults to `NZVM_REGISTRY_PATH`.
    smoothing : bool, optional
        Whether to apply smoothing to the model. Defaults to False.

    Returns
    -------
    None
    """
    import time

    start_time = time.time()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    out_dir = out_dir.resolve()
    out_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Using vm_params file: {vm_params_path}")
    with open(vm_params_path, "r") as f:
        vm_params: Dict = yaml.safe_load(f)

    cvm_registry = CVMRegistry(vm_params["model_version"], nzvm_registry, logger=logger)
    model_extent = ModelExtent(vm_params)
    global_mesh = gen_full_model_grid_great_circle(model_extent, logger)
    write_velo_mod_corners_text_file(global_mesh, str(out_dir), logger)

    velo_mod_1d_data, nz_tomography_data, global_surfaces, basin_data_list = (
        cvm_registry.load_all_global_data(logger)
    )

    in_basin_mesh, partial_global_mesh_list = (
        InBasinGlobalMesh.preprocess_basin_membership(
            global_mesh,
            basin_data_list,
            logger,
            smooth_bound=nz_tomography_data.smooth_boundary,
        )
    )

    for j in range(len(global_mesh.y)):
        logger.info(
            f"Generating velocity model {j * 100 / len(global_mesh.y):.2f}% complete."
        )
        partial_global_mesh = partial_global_mesh_list[j]
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

            basin_indices = in_basin_mesh.get_basin_membership(k, j)
            in_basin_list = [
                InBasin(basin_data, len(global_mesh.z))
                for basin_data in basin_data_list
            ]
            for basin_idx in basin_indices:
                if basin_idx >= 0:
                    in_basin_list[basin_idx].in_basin_lat_lon = True

            if smoothing:
                pass  # Add smoothing logic here if needed
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
                    in_basin_mesh,
                    vm_params["topo_type"],
                    logger,
                )

                partial_global_qualities.rho[k] = qualities_vector.rho
                partial_global_qualities.vp[k] = qualities_vector.vp
                partial_global_qualities.vs[k] = qualities_vector.vs
                partial_global_qualities.inbasin[k] = qualities_vector.inbasin

        write_global_qualities(
            out_dir, partial_global_mesh, partial_global_qualities, vm_params, j, logger
        )

    logger.info("Generation of velocity model 100% complete.")
    logger.info("Model generation complete.")

    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds.")


@cli.from_docstring(app)
def empty_command() -> None:
    """
    Empty command for testing purposes.

    Returns
    -------
    None
    """
    pass


if __name__ == "__main__":
    app()
