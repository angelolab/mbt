from __future__ import annotations

from os import PathLike
from typing import Literal

import ngff_zarr as nz
import xarray as xr
import zarr
import zarrs  # noqa: F401
from loguru import logger
from upath import UPath

from mbt.core.models import C, X, Y

# --- Zarr Configuration ---


def _configure_zarr() -> None:
    """Sets the global zarr configuration for optimized writing."""
    zarr.config.set(
        {
            "threading.max_workers": None,
            "array.write_empty_chunks": False,
            "codec_pipeline": {
                "path": "zarrs.ZarrsCodecPipeline",
                "validate_checksums": True,
                "store_empty_chunks": False,
                "chunk_concurrent_maximum": None,
                "chunk_concurrent_minimum": 4,
                "batch_size": 1,
            },
        }
    )


# --- Helper Functions ---


def _calculate_auto_shards(n_masses: int, ny: int, nx: int, storage_chunks: dict[str, int]) -> dict[str, int]:
    """Calculate default chunks_per_shard based on OME-Zarr strategy.

    Aims to create roughly one shard file per image/variable by specifying
    the number of storage chunks along each dimension to group into a shard.
    """
    base_chunk_c = storage_chunks.get(C, 1)
    base_chunk_y = storage_chunks.get(Y, ny)
    base_chunk_x = storage_chunks.get(X, nx)

    # Ensure at least 1 chunk per shard dimension
    shard_c = max(1, n_masses // base_chunk_c)
    shard_y = max(1, ny // base_chunk_y)
    shard_x = max(1, nx // base_chunk_x)

    return {C: shard_c, Y: shard_y, X: shard_x}


def _calculate_auto_shard_shape(n_masses: int, ny: int, nx: int) -> dict[str, int]:
    """Calculate default shard_shape to encompass the entire dataset variable."""
    return {C: n_masses, Y: ny, X: nx}


# --- OME-Zarr Helper ---


def _dataarray_to_multiscales(
    img_data: xr.DataArray,
    fov_microns: float,
    ny: int,
    nx: int,
    chunks: dict[str, int],
    scale_factors: list[dict[str, int]] | None = None,
):
    """Converts an Xarray DataArray to NGFF multiscales representation."""
    try:
        import ngff_zarr as nz
    except ImportError as e:
        raise ImportError("ngff-zarr is required for OME-Zarr writing. Install with mbt[zarr]") from e

    if scale_factors is None:
        scale_factors = []

    ngff_image = nz.to_ngff_image(
        data=img_data,
        dims=[C, Y, X],
        scale={
            C: 1,
            Y: fov_microns / ny,
            X: fov_microns / nx,
        },
        translation=None,
        name=str(img_data.name),
        axes_units={X: "micrometer", Y: "micrometer"},
    )

    multiscales = nz.to_multiscales(
        ngff_image,
        scale_factors=scale_factors,
        chunks=chunks,
    )

    # Extract channel names safely
    channel_names = []
    if "c" in img_data.coords:
        for channel_name in img_data.coords["c"]:
            name = str(channel_name.item()) if hasattr(channel_name, "item") else str(channel_name)
            channel_names.append(name)

    multiscales.metadata = nz.Metadata(
        axes=[
            nz.Axis(name=C, type="channel"),
            nz.Axis(name=Y, type="space", unit="micrometer"),
            nz.Axis(name=X, type="space", unit="micrometer"),
        ],
        datasets=[
            nz.Dataset(
                path=str(img_data.name),
                coordinateTransformations=[
                    nz.Scale(scale=[1, fov_microns / ny, fov_microns / nx]),
                    nz.Translation(translation=[0, 0, 0]),
                ],
            )
        ],
        coordinateTransformations=[
            nz.Scale(scale=[1, fov_microns / ny, fov_microns / nx]),
            nz.Translation(translation=[0, 0, 0]),
        ],
        omero=(
            nz.Omero(
                channels=[
                    nz.OmeroChannel(
                        color="FFFFFF",  # Default to white
                        window=nz.OmeroWindow(min=0, max=0, start=0, end=0),
                        label=label,
                    )
                    for label in channel_names
                ]
            )
            if channel_names
            else None
        ),
        name=str(img_data.name),
        version="0.5",
    )
    return multiscales


# --- Public Write Functions ---


def write_dataset_to_zarr(
    dataset: xr.Dataset,
    output_directory: PathLike,
    run_fmt_name: str,
    fov_fmt_name: str,
    n_masses: int,
    ny: int,
    nx: int,
    storage_chunks: dict[str, int],
    shard_shape: dict[str, int] | Literal["auto"] | None = "auto",
) -> None:
    """Write an Xarray Dataset to a Zarr store.

    Parameters
    ----------
    dataset
        The Xarray Dataset containing image data (e.g., counts, intensity).
    output_directory
        Path to the base output directory. A subdirectory named `run_fmt_name`
        will be created.
    run_fmt_name
        Formatted run name used for the subdirectory.
    fov_fmt_name
        Formatted FOV name used for the Zarr filename.
    n_masses
        Number of masses/channels, used for automatic chunking if needed.
    ny
        Number of pixels in the Y dimension.
    nx
        Number of pixels in the X dimension.
    storage_chunks
        Chunking configuration for zarr shards. If 'auto', calculates based
        on dimensions.
    shard_shape
        Desired total shape of each shard file in pixels/voxels. If 'auto',
        defaults to the full variable shape (n_masses, ny, nx).
    """
    _configure_zarr()

    out_dir = UPath(output_directory) / run_fmt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{fov_fmt_name}.zarr"

    # Calculate final shard shape if set to auto or None
    if shard_shape == "auto" or shard_shape is None:
        calculated_shard_shape = _calculate_auto_shard_shape(n_masses, ny, nx)
    else:
        calculated_shard_shape = shard_shape  # Use user-provided shape

    storage_chunks_tuple = tuple(storage_chunks[d] for d in [C, Y, X])
    # Ensure calculated_shard_shape is ordered correctly
    shard_shape_tuple = tuple(calculated_shard_shape[d] for d in [C, Y, X])
    print(f"Storage Chunks: {storage_chunks_tuple}, Shard Shape: {shard_shape_tuple}")  # Updated print

    dataset.to_zarr(
        output_path,
        consolidated=True,
        mode="w",
        encoding={
            v: {
                "chunks": storage_chunks_tuple,
                "shards": shard_shape_tuple,  # Pass the calculated shard shape
            }
            for v in dataset.data_vars
        },
    )
    logger.info(f"Dataset Zarr written to {output_path}.")


def write_ome_zarr(
    image_data: dict[str, xr.DataArray],
    output_directory: PathLike,
    run_fmt_name: str,
    fov_fmt_name: str,
    ny: int,
    nx: int,
    n_masses: int,
    fov_microns: float,
    image_type: list[Literal["counts", "intensity", "intensity_width"]] | str | list[str] | None = None,
    scale_factors: list[dict[str, int]] | None = None,
    chunks_per_shard: dict[str, int] | Literal["auto"] | None = "auto",
) -> None:
    """Write specified image types to individual OME-Zarr stores.

    Parameters
    ----------
    image_data
        Dictionary mapping image type names (e.g., 'counts') to their
        corresponding Xarray DataArrays.
    output_directory
        Path to the base output directory. A subdirectory named `run_fmt_name`
        will be created.
    run_fmt_name
        Formatted run name used for the subdirectory.
    fov_fmt_name
        Formatted FOV name used for the OME-Zarr filenames.
    ny
        Number of pixels in the Y dimension.
    nx
        Number of pixels in the X dimension.
    n_masses
        Number of masses/channels, used for automatic chunking if needed.
    fov_microns
        Size of the field of view in microns.
    image_type
        Image types to write. Defaults to all types found in `image_data`.
    scale_factors
        List of scale factors for multiscale representation. Defaults to `None`.
    chunks_per_shard
        Chunking configuration for zarr shards. If 'auto', calculates based
        on dimensions.

    Raises
    ------
    ImportError
        If ngff-zarr is not installed.
    ValueError
        If an invalid image type is requested (not present in `image_data`).
    """
    _configure_zarr()

    out_dir = UPath(output_directory) / fov_fmt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    available_types = set(image_data.keys())
    if image_type is None:
        types_to_write = list(available_types)
    elif isinstance(image_type, str):
        types_to_write = [image_type]
    else:
        types_to_write = image_type

    # Validate requested types
    requested_types = set(types_to_write)
    if not requested_types.issubset(available_types):
        invalid = requested_types - available_types
        raise ValueError(f"Invalid image_type requested: {invalid}. Available types are: {available_types}")

    # Define base storage chunks
    storage_chunks = {C: 1, Y: min(ny, 256), X: min(nx, 256)}

    # Calculate shards if set to auto or None
    if chunks_per_shard == "auto" or chunks_per_shard is None:
        chunks_per_shard = _calculate_auto_shards(n_masses, ny, nx, storage_chunks)

    for img_type in types_to_write:
        img_da = image_data[img_type]
        output_path = out_dir / f"{fov_fmt_name}-{img_type}.ome.zarr"

        multiscales = _dataarray_to_multiscales(
            img_da,
            fov_microns=fov_microns,
            ny=ny,
            nx=nx,
            chunks=storage_chunks,
            scale_factors=scale_factors,
        )

        nz.to_ngff_zarr(
            output_path,
            multiscales=multiscales,
            version="0.5",
            overwrite=True,
            chunks_per_shard=chunks_per_shard,
        )
        logger.info(f"OME-Zarr for '{img_type}' written to {output_path}.")
