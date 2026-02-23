"""Compress velocity models for archival outputs"""

from pathlib import Path
from typing import Annotated

import h5py
import numpy as np
import typer
import xarray as xr

app = typer.Typer()


def get_extrema(h5_dataset: h5py.Dataset) -> tuple[float, float]:
    """Get extreme values of hdf5 dataset.

    Parameters
    ----------
    h5_dataset : h5py.Dataset
        HDF5 dataset to find extrema for.

    Returns
    -------
    tuple[float, float]
        (min, max) values for dataset.
    """
    min_v, max_v = np.inf, -np.inf

    for i in range(h5_dataset.shape[1]):
        slice_data = h5_dataset[:, i, :]
        min_v = min(min_v, np.nanmin(slice_data))
        max_v = max(max_v, np.nanmax(slice_data))
    return min_v, max_v


def compress_quality(file: h5py.File, quality: str) -> xr.DataArray:
    """Compress velocity model quality using quantisation method.

    Parameters
    ----------
    file : h5py.File
        File to read quality from.
    quality : str
        Quality to read, e.g. rho.

    Returns
    -------
    xr.DataArray
        A quantised dataarray with uint8 values. The scale factor, and
        add offset attributes record the scale of the quantised array
        and minimum value, respectively.
    """
    quality_array = file["properties"][quality]
    shape = quality_array.shape
    (nz, ny, nx) = shape
    quantised_array = np.zeros(shape, dtype=np.uint8)
    int_max = np.iinfo(np.uint8).max
    min, max = get_extrema(quality_array)
    scale = (max - min) / int_max
    for i in range(ny):
        # Copy out one y-slice to a local copy.
        y_slice = quality_array[:, i, :].astype(np.float32)
        # y_slice_quantised = round(y_slice / scale_max) as uint8
        # Need to do this with `out` parameters to avoid extra unneeded copies
        np.subtract(y_slice, min, out=y_slice)
        np.divide(y_slice, scale, out=y_slice)
        np.round(y_slice, out=y_slice)
        y_slice_quantised = y_slice.astype(np.uint8)
        quantised_array[:, i, :] = y_slice_quantised

    attrs = dict(quality_array.attrs)
    attrs["scale_factor"] = scale
    attrs["add_offset"] = min
    attrs["_FillValue"] = max
    z = np.arange(nz)
    y = np.arange(ny)
    x = np.arange(nx)

    da = xr.DataArray(
        quantised_array,
        dims=("z", "y", "x"),
        coords=dict(z=z, y=y, x=x),
        attrs=attrs,
    )
    return da


def read_inbasin(file: h5py.File) -> xr.DataArray:
    """Read inbasin vector from velocity model.

    Parameters
    ----------
    file : h5py.File
        Velocity model to read from.

    Returns
    -------
    xr.DataArray
        Data array for inbasin quality.
    """
    inbasin = np.array(file["properties"]["inbasin"])
    (nz, ny, nx) = inbasin.shape
    z = np.arange(nz)
    y = np.arange(ny)
    x = np.arange(nx)
    attrs = dict(file["properties"]["inbasin"].attrs)
    da = xr.DataArray(
        inbasin, dims=("z", "y", "x"), coords=dict(z=z, y=y, x=x), attrs=attrs
    )
    return da


def compressed_vm_as_dataset(file: h5py.File) -> xr.Dataset:
    """Convert an HDF5 velocity model into a compressed and quantised xarray dataset.

    Parameters
    ----------
    file : h5py.File
        Velocity model to quantise.

    Returns
    -------
    xr.Dataset
        Compressed and quantised dataset representing the read
        velocity model.
    """
    compressed_vp = compress_quality(file, "vp")
    compressed_vs = compress_quality(file, "vs")
    compressed_rho = compress_quality(file, "rho")
    inbasin = read_inbasin(file)
    lat = np.array(file["mesh"]["lat"])
    lon = np.array(file["mesh"]["lon"])

    z_resolution = float(file["config"].attrs["h_depth"])
    nz = compressed_vp.shape[0]
    z = np.arange(nz) * z_resolution

    ds = xr.Dataset(
        {
            "vp": compressed_vp,
            "vs": compressed_vs,
            "rho": compressed_rho,
            "inbasin": inbasin,
        },
    )
    ds.attrs.update(file["config"].attrs)
    # Now that the dimensions of the above arrays are consolidated, we can re-use them for the inbasin assignment.

    ds = ds.assign_coords(
        dict(lon=(("x", "y"), lon), lat=(("x", "y"), lat), depth=(("z"), z)),
    )
    ds = ds.set_coords(["lat", "lon", "depth"])
    return ds


@app.command()
def compress_vm(
    vm_path: Path,
    output: Path,
    complevel: Annotated[int, typer.Option(min=1, max=19)] = 4,
    chunk_x: Annotated[int | None, typer.Option(min=1)] = 64,
    chunk_y: Annotated[int | None, typer.Option(min=1)] = 256,
    chunk_z: Annotated[int | None, typer.Option(min=1)] = 256,
    shuffle: bool = True,
) -> None:
    """Compress a velocity model for archival storage.

    Parameters
    ----------
    vm_path : Path
        Path to velocity model to compress.
    output : Path
        Path to store compressed velocity model.
    complevel : int
        Compression level for zlib compression.
    chunk_x : int | None, optional
        Chunksize in x direction. Set to ``None`` to infer dataset sive.
    chunk_y : int | None, optional
        Chunksize in y direction. Set to ``None`` to infer dataset size.
    chunk_z : int | None, optional
        Chunksize in z direction. Set to ``None`` to infer dataset size.
    shuffle : bool
        If set, enable bit-level shuffling for dataset compression.
    """
    with h5py.File(vm_path) as vm:
        dset = compressed_vm_as_dataset(vm)

    common_options = dict(
        dtype="uint8",
        zlib=True,
        complevel=complevel,
        shuffle=shuffle,
        chunksizes=(
            chunk_x or dset.sizes["x"],
            chunk_y or dset.sizes["y"],
            chunk_z or dset.sizes["z"],
        ),
    )

    dset.to_netcdf(
        output,
        encoding={
            "vp": common_options,
            "vs": common_options,
            "rho": common_options,
            "inbasin": common_options,
        },
        engine="h5netcdf",
    )
