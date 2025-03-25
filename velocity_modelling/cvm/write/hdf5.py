"""
Module for writing velocity model data in HDF5 format.

This module provides functions to write the velocity model data to HDF5 format files.
HDF5 (Hierarchical Data Format version 5) is an efficient binary format for storing large
scientific datasets with metadata.
"""

import datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from velocity_modelling.cvm.geometry import PartialGlobalMesh
from velocity_modelling.cvm.velocity3d import PartialGlobalQualities


def create_xdmf_file(hdf5_file: Path, vm_params: dict, logger: logging.Logger) -> None:
    """
    Create an XDMF file to make the HDF5 file compatible with ParaView.

    Parameters
    ----------
    hdf5_file : Path
        Path to the HDF5 file.
    vm_params : dict
        Dictionary containing model parameters.
    logger : logging.Logger
        Logger for reporting progress and errors.
    """
    xdmf_file = hdf5_file.with_suffix(".xdmf")

    nx = vm_params.get("nx")
    ny = vm_params.get("ny")
    nz = vm_params.get("nz")

    # Get the relative path to the HDF5 file from the XDMF file
    hdf5_relative = hdf5_file.name

    xdmf_content = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="Velocity_Model" GridType="Uniform">
      <Topology TopologyType="3DRectMesh" Dimensions="{nz} {ny} {nx}"/>
      <Geometry GeometryType="VXVYVZ">
        <DataItem Dimensions="{nx}" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_relative}:/mesh/x
        </DataItem>
        <DataItem Dimensions="{ny}" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_relative}:/mesh/y
        </DataItem>
        <DataItem Dimensions="{nz}" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_relative}:/mesh/z
        </DataItem>
      </Geometry>
      <Attribute Name="P-wave Velocity" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{nx} {ny} {nz}" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_relative}:/properties/vp
        </DataItem>
      </Attribute>
      <Attribute Name="S-wave Velocity" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{nx} {ny} {nz}" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_relative}:/properties/vs
        </DataItem>
      </Attribute>
      <Attribute Name="Density" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{nx} {ny} {nz}" NumberType="Float" Precision="4" Format="HDF">
          {hdf5_relative}:/properties/rho
        </DataItem>
      </Attribute>
      <Attribute Name="Basin Membership" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{nx} {ny} {nz}" NumberType="Int" Precision="1" Format="HDF">
          {hdf5_relative}:/properties/inbasin
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""

    try:
        with open(xdmf_file, "w") as f:
            f.write(xdmf_content)
        logger.log(logging.INFO, f"Created ParaView-compatible XDMF file: {xdmf_file}")
    except Exception as e:
        logger.log(logging.ERROR, f"Failed to create XDMF file: {e}")


def write_global_qualities(
    out_dir: Path,
    partial_global_mesh: PartialGlobalMesh,
    partial_global_qualities: PartialGlobalQualities,
    lat_ind: int,
    vm_params: dict,
    logger: Optional[Logger] = None,
) -> None:
    """
    Write a latitude slice of velocity data to a single consolidated HDF5 file.

    For the first slice (lat_ind=0), this creates the file and initializes the structure.
    For subsequent slices, it adds data to the existing file.

    Parameters
    ----------
    out_dir : Path
        Path to the output directory where the HDF5 file will be written.
    partial_global_mesh : PartialGlobalMesh
        Mesh data for the current latitude slice.
    partial_global_qualities : PartialGlobalQualities
        Velocity and density data for the current latitude slice.
    lat_ind : int
        Latitude index to determine the write mode (write or append).
    vm_params : dict
        Dictionary containing velocity model parameters from nzvm.cfg.
    logger : Logger, optional
        Logger instance for logging messages.

    Raises
    ------
    OSError
        If the HDF5 file cannot be written due to permissions or disk issues.
    """
    # Output filename - single file for all slices
    hdf5_file = Path(out_dir) / "velocity_model.h5"

    # Apply minimum VS constraint
    min_vs = vm_params.get(
        "min_vs", 0.0
    )  # Get the minimum Vs value from the parameters
    vs_data = np.copy(partial_global_qualities.vs)
    vs_data[vs_data < min_vs] = min_vs

    ny = vm_params.get(
        "ny"
    )  # This is a slice along y (=lat) axis, so partial_global_mesh has only one y value

    # If first slice, create the file and initialize structure
    if lat_ind == 0:
        try:
            with h5py.File(hdf5_file, "w") as f:
                logger.log(logging.DEBUG, f"Creating new HDF5 file: {hdf5_file}")

                # Add basic metadata attributes
                f.attrs["total_y_slices"] = ny
                f.attrs["format_version"] = "1.0"
                f.attrs["creation_date"] = datetime.datetime.now().isoformat()

                # Add all velocity model parameters from nzvm.cfg as root attributes
                if vm_params:
                    config_group = f.create_group("config")
                    for key, value in vm_params.items():
                        # Handle special case for enum types
                        if hasattr(value, "name"):
                            config_group.attrs[key] = value.name
                        elif hasattr(value, "value"):
                            config_group.attrs[key] = value.value
                        else:
                            try:
                                config_group.attrs[key] = value
                            except TypeError:
                                # If attribute can't be stored directly, convert to string
                                config_group.attrs[key] = str(value)

                    # Add a string representation of the original config for reference
                    config_lines = []
                    for key, value in vm_params.items():
                        if hasattr(value, "name"):
                            config_lines.append(f"{key.upper()}={value.name}")
                        else:
                            config_lines.append(f"{key.upper()}={value}")
                    config_str = "\n".join(config_lines)
                    config_group.attrs["config_string"] = config_str

                # Create groups for mesh and properties
                f.create_group("mesh")
                f.create_group("properties")

                # Create resizable datasets for properties
                props_group = f["properties"]
                nx, nz = partial_global_qualities.vp.shape

                # Create datasets with compression
                props_group.create_dataset(
                    "vp", shape=(nx, ny, nz), dtype="f4", compression="gzip"
                )
                props_group.create_dataset(
                    "vs", shape=(nx, ny, nz), dtype="f4", compression="gzip"
                )
                props_group.create_dataset(
                    "rho", shape=(nx, ny, nz), dtype="f4", compression="gzip"
                )
                props_group.create_dataset(
                    "inbasin", shape=(nx, ny, nz), dtype="i1", compression="gzip"
                )

                # Add metadata
                props_group["vp"].attrs["units"] = "km/s"
                props_group["vs"].attrs["units"] = "km/s"
                props_group["rho"].attrs["units"] = "g/cm³"
                props_group["vs"].attrs["min_value_enforced"] = min_vs

        except OSError as e:
            logger.log(logging.ERROR, f"Failed to create HDF5 file {hdf5_file}: {e}")
            raise OSError(f"Failed to create HDF5 file {hdf5_file}: {str(e)}")

    # Write this slice's data to the file
    try:
        with h5py.File(hdf5_file, "r+") as f:
            # Write mesh data only once (when completing the model)
            if lat_ind == ny - 1:
                # Now that we have all slices, write the complete mesh data
                mesh_group = f["mesh"]
                mesh_group.create_dataset("x", data=partial_global_mesh.x)
                # Create y as a linear space from -y to y, as partial_global_mesh.y is a scalar, has no range of all y values
                mesh_group.create_dataset(
                    "y",
                    data=np.linspace(
                        -1 * partial_global_mesh.y, partial_global_mesh.y, ny
                    ),
                )
                mesh_group.create_dataset("z", data=partial_global_mesh.z)
                mesh_group.create_dataset("lon", data=partial_global_mesh.lon)
                mesh_group.create_dataset("lat", data=partial_global_mesh.lat)

                # Mark file as complete
                f.attrs["complete"] = True

            # Write property data for this slice
            f["properties/vp"][:, lat_ind, :] = partial_global_qualities.vp
            f["properties/vs"][:, lat_ind, :] = vs_data
            f["properties/rho"][:, lat_ind, :] = partial_global_qualities.rho
            f["properties/inbasin"][:, lat_ind, :] = partial_global_qualities.inbasin

            # Update progress attribute
            f.attrs["last_slice_written"] = lat_ind

        logger.log(logging.DEBUG, f"Written slice {lat_ind} to {hdf5_file}")
    except OSError as e:
        logger.log(logging.ERROR, f"Failed to update HDF5 file {hdf5_file}: {e}")
        raise OSError(f"Failed to update HDF5 file {hdf5_file}: {str(e)}")
    except Exception as e:
        if isinstance(e, (SystemExit, KeyboardInterrupt)):
            raise  # Re-raise critical exceptions
        logger.log(logging.ERROR, f"Error writing to HDF5 file {hdf5_file}: {e}")
        raise RuntimeError(f"Error writing to HDF5 file {hdf5_file}: {str(e)}")

    # After writing all slices, create the XDMF file
    if lat_ind == vm_params.get("ny", 0) - 1:
        # Now that we have written the entire model, create the XDMF file
        create_xdmf_file(hdf5_file, vm_params, logger)
        logger.log(logging.INFO, "HDF5 model complete with ParaView compatibility")
